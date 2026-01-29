"""Unified training script for deep learning baseline models."""

import argparse
import json
import logging
import os
import random
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Libs.config import (load_dataset_config, load_experiment_config,
                         load_model_config)
from Libs.data.dataloader import load_arrows_data
from Libs.data.label_generator import FatigueLabelGenerator
from Libs.data.physics_processor import PhysicsModel
from Libs.data.timeseries_dataset import prepare_timeseries_data
from Libs.models.modules.contrastive_learning import (ContrastiveEncoder,
                                                      SimCLRTrainer,
                                                      TimeSeriesAugmentation)
from Libs.models.networks.attention_lstm_model import AttentionLSTMWrapper
from Libs.models.networks.cnn_lstm_model import CNNLSTMWrapper
from Libs.models.networks.gru_model import GRUWrapper
from Libs.models.networks.lstm_model import LSTMWrapper
from Libs.models.networks.tcn_model import TCNWrapper
from Libs.models.networks.transformer_model import TransformerWrapper


def _resolve_device(preferred: str) -> str:
    """Returns a runtime-safe compute device string based on availability."""
    if preferred == 'cuda' and not __import__('torch').cuda.is_available():
        return 'cpu'
    return preferred


def _set_seed(seed: Optional[int]) -> None:
    """Sets global random seeds for reproducibility."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes standard regression metrics and R² on RUL>0 subset."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = np.nan if ss_tot == 0 else r2_score(y_true, y_pred)
    
    if ss_tot == 0:
        print(f"[Warning] _regression_metrics: R² cannot be computed (SS_tot=0, all labels are identical: {np.mean(y_true):.4f})")
    
    mask = y_true > 0
    n_pos = int(np.sum(mask))
    if n_pos >= 2:
        ss_tot_pos = np.sum((y_true[mask] - np.mean(y_true[mask])) ** 2)
        r2_rul_positive = np.nan if ss_tot_pos == 0 else float(r2_score(y_true[mask], y_pred[mask]))
    else:
        r2_rul_positive = np.nan
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'r2_rul_positive': r2_rul_positive}


def _aggregate_by_window(
    preds: np.ndarray,
    targets: np.ndarray,
    sample_to_window: Optional[np.ndarray],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Aggregates sample-level predictions/targets to window-level using mean pooling.
    """
    if sample_to_window is None:
        return None
    window_to_preds: Dict[int, list] = {}
    window_to_targets: Dict[int, float] = {}
    for idx, win_idx in enumerate(sample_to_window):
        window_to_preds.setdefault(int(win_idx), []).append(float(preds[idx]))
        window_to_targets[int(win_idx)] = float(targets[idx])
    window_indices = sorted(window_to_preds.keys())
    agg_preds = np.array([np.mean(window_to_preds[w]) for w in window_indices], dtype=np.float32)
    agg_targets = np.array([window_to_targets[w] for w in window_indices], dtype=np.float32)
    return agg_preds, agg_targets


def _build_label_scaler(config: Dict[str, Any]):
    """Returns (encode_fn, decode_fn, encode_using_state_fn).
    encode_fn(arr) -> (encoded, state). decode_fn(arr, state) -> decoded.
    encode_using_state_fn(arr, state) -> encoded; use for val/test with train state.
    """
    enable = config.get('enable', False)
    method = config.get('method', 'standard')
    if not enable:
        def _enc(x):
            return (x, {'enabled': False})
        def _dec(x, s):
            return x
        def _enc_use(arr, s):
            return np.asarray(arr, dtype=np.float32) if not isinstance(arr, np.ndarray) else arr
        return _enc, _dec, _enc_use

    if method == 'standard':
        def encode(arr: np.ndarray):
            mean = float(np.mean(arr))
            std = float(np.std(arr)) if np.std(arr) > 0 else 1.0
            return (arr - mean) / std, {'method': method, 'mean': mean, 'std': std, 'enabled': True}
        def decode(arr: np.ndarray, state: Dict[str, Any]):
            return arr * state['std'] + state['mean']
        def encode_using_state(arr: np.ndarray, state: Dict[str, Any]):
            if not state.get('enabled', True):
                return np.asarray(arr, dtype=np.float32)
            return (np.asarray(arr, dtype=np.float32) - state['mean']) / state['std']
    elif method == 'minmax':
        def encode(arr: np.ndarray):
            mn = float(np.min(arr))
            mx = float(np.max(arr))
            scale = mx - mn if mx > mn else 1.0
            return (arr - mn) / scale, {'method': method, 'min': mn, 'max': mx, 'scale': scale, 'enabled': True}
        def decode(arr: np.ndarray, state: Dict[str, Any]):
            return arr * state['scale'] + state['min']
        def encode_using_state(arr: np.ndarray, state: Dict[str, Any]):
            if not state.get('enabled', True):
                return np.asarray(arr, dtype=np.float32)
            return (np.asarray(arr, dtype=np.float32) - state['min']) / state['scale']
    elif method == 'robust':
        def encode(arr: np.ndarray):
            med = float(np.median(arr))
            iqr = float(np.percentile(arr, 75) - np.percentile(arr, 25))
            scale = iqr if iqr > 0 else 1.0
            return (arr - med) / scale, {'method': method, 'median': med, 'iqr': iqr, 'scale': scale, 'enabled': True}
        def decode(arr: np.ndarray, state: Dict[str, Any]):
            return arr * state['scale'] + state['median']
        def encode_using_state(arr: np.ndarray, state: Dict[str, Any]):
            if not state.get('enabled', True):
                return np.asarray(arr, dtype=np.float32)
            return (np.asarray(arr, dtype=np.float32) - state['median']) / state['scale']
    else:
        def _enc(x):
            return (x, {'enabled': False})
        def _dec(x, s):
            return x
        def _enc_use(arr, s):
            return np.asarray(arr, dtype=np.float32) if not isinstance(arr, np.ndarray) else arr
        return _enc, _dec, _enc_use

    def encode_with_state(arr: np.ndarray):
        encoded, state = encode(arr)
        return encoded, state

    return encode_with_state, decode, encode_using_state


def _build_criterion(loss_cfg: Dict[str, Any]):
    """Builds torch loss function based on configuration."""
    loss_type = loss_cfg.get('type', 'mse')
    delta = float(loss_cfg.get('huber_delta', 1.0))
    mapping = {
        'mse': torch.nn.MSELoss(reduction='none'),
        'huber': torch.nn.HuberLoss(reduction='none', delta=delta),
        'smooth_l1': torch.nn.SmoothL1Loss(reduction='none', beta=delta),
    }
    return mapping.get(loss_type, mapping['mse'])


def _get_warmup_lr(epoch: int, warmup_epochs: int, base_lr: float) -> float:
    """Computes learning rate with linear warmup.
    
    Args:
        epoch: Current epoch (0-indexed).
        warmup_epochs: Number of warmup epochs.
        base_lr: Base learning rate after warmup.
    
    Returns:
        Learning rate for current epoch.
    """
    if warmup_epochs == 0 or epoch >= warmup_epochs:
        return base_lr
    return base_lr * (epoch + 1) / warmup_epochs


def _build_sample_weights(labels: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    """Computes per-sample weights using simple positive/zero weighting."""
    if not cfg.get('enable', False):
        return np.ones_like(labels, dtype=np.float32)
    pos_w = float(cfg.get('positive_weight', 1.0))
    zero_w = float(cfg.get('zero_weight', 1.0))
    weights = np.where(labels > 0, pos_w, zero_w).astype(np.float32)
    return weights


def _split_data(
    time_windows: list,
    labels: np.ndarray,
    test_size: float,
    validation_size: float,
    random_state: Optional[int],
    split_method: str = "temporal",
    ensure_pre_failure_in_train: bool = True,
    min_pre_failure_in_test: Optional[int] = None,
) -> Tuple[list, list, list, np.ndarray, np.ndarray, np.ndarray]:
    """
    Partitions samples into train, validation, and test splits with balanced pre-failure/post-failure distribution.

    Args:
        time_windows: List of time window tuples.
        labels: Array of labels (RUL values).
        test_size: Fraction of samples for testing.
        validation_size: Fraction of samples for validation (from training set).
        random_state: Random seed for reproducibility.
        split_method: "temporal" for time-series split or "random" for random split.
        ensure_pre_failure_in_train: If True, ensures training set contains pre-failure data.
        min_pre_failure_in_test: If set, ensure test has at least this many pre-failure samples (temporal only).

    Returns:
        Tuple of (train_windows, val_windows, test_windows, train_labels, val_labels, test_labels).
    """
    indices = np.arange(len(time_windows))

    if split_method == "temporal":
        pre_failure_mask = labels > 0
        post_failure_mask = labels == 0
        pre_failure_indices = indices[pre_failure_mask]
        post_failure_indices = indices[post_failure_mask]

        n_pre = len(pre_failure_indices)
        n_post = len(post_failure_indices)
        n_total = len(indices)

        n_test = int(n_total * test_size)
        n_val = int((n_total - n_test) * validation_size)
        n_train = n_total - n_test - n_val

        n_pre_test = max(1, int(n_pre * test_size)) if n_pre > 0 else 0
        n_pre_val = max(1, int(n_pre * validation_size)) if n_pre > 0 else 0
        n_pre_train = n_pre - n_pre_test - n_pre_val

        if min_pre_failure_in_test is not None and n_pre >= min_pre_failure_in_test and n_pre_test < min_pre_failure_in_test:
            n_pre_test = min(min_pre_failure_in_test, n_pre)
            remainder = n_pre - n_pre_test
            n_pre_val = max(1, int(remainder * validation_size)) if remainder > 0 else 0
            n_pre_train = remainder - n_pre_val

        n_post_test = n_test - n_pre_test
        n_post_val = n_val - n_pre_val
        n_post_train = n_post - n_post_test - n_post_val

        train_idx_list = []
        val_idx_list = []
        test_idx_list = []

        if n_pre > 0:
            pre_train_end = n_pre_train
            pre_val_end = pre_train_end + n_pre_val

            train_idx_list.extend(pre_failure_indices[:pre_train_end])
            val_idx_list.extend(pre_failure_indices[pre_train_end:pre_val_end])
            test_idx_list.extend(pre_failure_indices[pre_val_end:])
        
        if n_post > 0:
            post_train_end = n_post_train
            post_val_end = post_train_end + n_post_val
            
            train_idx_list.extend(post_failure_indices[:post_train_end])
            val_idx_list.extend(post_failure_indices[post_train_end:post_val_end])
            test_idx_list.extend(post_failure_indices[post_val_end:])
        
        train_idx = np.array(train_idx_list, dtype=int)
        val_idx = np.array(val_idx_list, dtype=int)
        test_idx = np.array(test_idx_list, dtype=int)
        
        train_idx = np.sort(train_idx)
        val_idx = np.sort(val_idx)
        test_idx = np.sort(test_idx)
        
        pre_in_train = np.sum(labels[train_idx] > 0)
        pre_in_val = np.sum(labels[val_idx] > 0)
        pre_in_test = np.sum(labels[test_idx] > 0)
        
        print(f"[Info] _split_data: Balanced split - Train: {len(train_idx)} samples ({pre_in_train} pre-failure, {len(train_idx)-pre_in_train} post-failure), "
              f"Val: {len(val_idx)} samples ({pre_in_val} pre-failure, {len(val_idx)-pre_in_val} post-failure), "
              f"Test: {len(test_idx)} samples ({pre_in_test} pre-failure, {len(test_idx)-pre_in_test} post-failure)")
    else:
        train_val_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)
        
        if len(train_val_idx) > 0:
            relative_validation_size = validation_size / (1 - test_size)
            train_idx, val_idx = train_test_split(train_val_idx, test_size=relative_validation_size, random_state=random_state)
        else:
            train_idx, val_idx = [], []

    train_windows = [time_windows[i] for i in train_idx]
    val_windows = [time_windows[i] for i in val_idx]
    test_windows = [time_windows[i] for i in test_idx]
    
    train_labels = np.asarray([labels[i] for i in train_idx], dtype=np.float32)
    val_labels = np.asarray([labels[i] for i in val_idx], dtype=np.float32)
    test_labels = np.asarray([labels[i] for i in test_idx], dtype=np.float32)

    return train_windows, val_windows, test_windows, train_labels, val_labels, test_labels


def train_dl_baseline(
    model_type: str,
    data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    dataset_config: Optional[Dict[str, Any]] = None,
    model_config: Optional[Dict[str, Any]] = None,
    experiment_config: Optional[Dict[str, Any]] = None,
    prepared_data: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train a deep learning baseline model.
    
    Args:
        model_type: Type of model ('lstm', 'gru', 'transformer', 'cnn_lstm', 'tcn', 'attention_lstm')
        data_path: Path to raw data file
        output_dir: Directory to save model
        dataset_config: Dataset configuration dict
        model_config: Model configuration dict
        experiment_config: Experiment configuration dict
        
    Returns:
        Tuple of (model, metrics_dict)
    """
    if dataset_config is None:
        from Libs.config import load_dataset_config
        dataset_config = load_dataset_config()
    if model_config is None:
        from Libs.config import load_model_config
        model_config = load_model_config()
    if experiment_config is None:
        from Libs.config import load_experiment_config
        experiment_config = load_experiment_config()
    
    data_paths = dataset_config.get('data_paths', {})
    split_settings = dataset_config.get('preprocessing', {}).get('data_split', {})
    dataset_time_series = dataset_config.get('preprocessing', {}).get('time_series', {})
    exp_time_series = experiment_config.get('preprocessing', {}).get('time_series', {})
    time_series_settings = {**dataset_time_series, **exp_time_series}
    training_settings = experiment_config.get('training', {})
    model_paths = model_config.get('model_paths', {})
    
    model_settings = model_config.get('deep_learning_models', {}).get(model_type, {})
    
    raw_data_path = data_path if data_path is not None else data_paths.get('raw_data', 'Input/raw/data.mat')
    model_root = output_dir if output_dir is not None else model_paths.get('models_dir', 'Output/models')
    results_dir = model_paths.get('results_dir', 'Output/results')
    os.makedirs(model_root, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    test_size = float(split_settings.get('test_size', 0.2))
    validation_size = float(split_settings.get('validation_size', 0.2))
    random_state = split_settings.get('random_state', 42)
    split_method = split_settings.get('split_method', 'temporal')
    ensure_pre_failure_in_train = split_settings.get('ensure_pre_failure_in_train', True)
    min_pre_failure_in_test = split_settings.get('min_pre_failure_in_test')
    seed = experiment_config.get('random_state', split_settings.get('random_state', None))
    _set_seed(seed)
    
    epochs = int(training_settings.get('num_epochs', 100))
    learning_rate = float(training_settings.get('learning_rate', 0.001))
    batch_size = int(training_settings.get('batch_size', 32))
    loss_config = training_settings.get('loss_function', {})
    
    early_stopping_config = training_settings.get('early_stopping', {})
    early_stopping_config['enable'] = early_stopping_config.get('enable', True)
    scheduler_config = training_settings.get('scheduler', {})
    if scheduler_config.get('type') == 'reduce_on_plateau':
        reduce_config = scheduler_config.get('reduce_on_plateau', {})
        scheduler_config['factor'] = reduce_config.get('factor', 0.5)
        scheduler_config['patience'] = reduce_config.get('patience', 10)
        scheduler_config['threshold'] = reduce_config.get('threshold', 0.0001)
        scheduler_config['cooldown'] = reduce_config.get('cooldown', 0)
    
    preferred_device = model_settings.get('device', 'cuda')
    device = _resolve_device(preferred_device)
    
    logger = logging.getLogger(__name__)
    
    logger.info('=' * 60)
    logger.info(f'Training {model_type.upper()} Model')
    logger.info('=' * 60)
    
    if prepared_data is None:
        logger.info('[1/5] Loading ARROWS data...')
        data = load_arrows_data(raw_data_path)
        if data is None:
            raise RuntimeError('Failed to load raw dataset.')
        
        logger.info('[2/5] Computing elastic modulus...')
        physics = PhysicsModel(dataset_config)
        modulus_result = physics.compute_elastic_modulus(data['sensors'], data['time'])
        
        logger.info('[3/5] Building training dataframe...')
        label_generator = FatigueLabelGenerator(dataset_config)
        training_df = label_generator.prepare_training_data(
            modulus_result, data['sensors'], data['time']
        )
        
        logger.info('[4/5] Preparing time series data...')
        cycle_indices = modulus_result.get('cycle_indices', [])
        labels = training_df['rul'].to_numpy()
        
        total_labels = len(labels)
        nan_labels_count = np.sum(np.isnan(labels))
        valid_labels_count = total_labels - nan_labels_count
        logger.info(f"[Diagnostic] train_dl_baseline: {total_labels} total labels, {nan_labels_count} NaN labels, {valid_labels_count} valid labels")
        
        valid_mask = ~np.isnan(labels)
        valid_indices = np.where(valid_mask)[0]
        
        time_windows = []
        target_values = []
        skipped_indices = 0
        for index in valid_indices:
            if index < len(cycle_indices):
                time_windows.append(cycle_indices[index])
                target_values.append(float(labels[index]))
            else:
                skipped_indices += 1
        
        logger.info(f"[Diagnostic] train_dl_baseline: {len(valid_indices)} valid indices, {skipped_indices} skipped (index >= len(cycle_indices)), {len(time_windows)} time windows created")
        
        train_windows, val_windows, test_windows, train_labels, val_labels, test_labels = _split_data(
            time_windows, np.array(target_values), test_size, validation_size, random_state,
            split_method=split_method, ensure_pre_failure_in_train=ensure_pre_failure_in_train,
            min_pre_failure_in_test=min_pre_failure_in_test,
        )

        oversample_times = int(split_settings.get('oversample_pre_failure_times', 1))
        if oversample_times >= 2:
            new_tw, new_tl = [], []
            for i in range(len(train_windows)):
                new_tw.append(train_windows[i])
                new_tl.append(float(train_labels[i]))
                if train_labels[i] > 0:
                    for _ in range(oversample_times - 1):
                        new_tw.append(train_windows[i])
                        new_tl.append(float(train_labels[i]))
            train_windows = new_tw
            train_labels = np.array(new_tl, dtype=np.float32)
        
        train_pre_only = bool(split_settings.get('train_pre_only', False))
        if train_pre_only:
            mask = train_labels > 0
            train_windows = [train_windows[i] for i in range(len(train_windows)) if mask[i]]
            train_labels = train_labels[mask]
            if len(train_windows) == 0:
                raise ValueError("train_pre_only=true and no RUL>0 in train.")
        
        num_sensors = len([k for k in data['sensors'].keys()])
        input_dim = num_sensors * 2
        
        max_sequence_length = time_series_settings.get('max_sequence_length', None)
        if max_sequence_length is not None:
            max_sequence_length = int(max_sequence_length)
            print(f'Using max_sequence_length: {max_sequence_length} (sequences longer than this will be uniformly sampled)')
        sequence_length = time_series_settings.get('sequence_length', None)
        if sequence_length is not None:
            sequence_length = int(sequence_length)
        stride = time_series_settings.get('stride', None)
        if stride is not None:
            stride = int(stride)
        use_full_sequence = bool(time_series_settings.get('use_full_sequence', False))
        
        augmentation_config = dataset_config.get('augmentation', {})
        
        common_ts_kwargs = dict(
            device="cpu",
            max_sequence_length=max_sequence_length,
            sequence_length=sequence_length,
            stride=stride,
            use_full_sequence=use_full_sequence,
            augmentation=augmentation_config,
        )
        
        label_norm_cfg = experiment_config.get('preprocessing', {}).get('label_normalization', {})
        encode_label, decode_label_base, encode_using_state = _build_label_scaler(label_norm_cfg)

        train_labels_encoded, scaler_state = encode_label(np.array(train_labels, dtype=np.float32))
        val_labels_encoded = encode_using_state(np.array(val_labels, dtype=np.float32), scaler_state)
        test_labels_encoded = encode_using_state(np.array(test_labels, dtype=np.float32), scaler_state)
        
        decode_label = lambda x, s=scaler_state: decode_label_base(x, s)
        
        sample_weight_cfg = loss_config.get('sample_weighting', {})
        train_sample_weights = _build_sample_weights(np.array(train_labels, dtype=np.float32), sample_weight_cfg)
        val_sample_weights = _build_sample_weights(np.array(val_labels, dtype=np.float32), sample_weight_cfg)
        test_sample_weights = _build_sample_weights(np.array(test_labels, dtype=np.float32), sample_weight_cfg)
        
        train_data = prepare_timeseries_data(
            data['sensors'], train_windows, train_labels_encoded, sample_weights=train_sample_weights, **common_ts_kwargs
        )
        val_data = prepare_timeseries_data(
            data['sensors'], val_windows, val_labels_encoded, sample_weights=val_sample_weights, **common_ts_kwargs
        )
        test_data = prepare_timeseries_data(
            data['sensors'], test_windows, test_labels_encoded, sample_weights=test_sample_weights, **common_ts_kwargs
        )
        
        if len(train_data) == 0:
            msg = (
                f"No training samples generated (labels={total_labels}, "
                f"valid_labels={valid_labels_count}, time_windows={len(time_windows)})."
            )
            raise ValueError(msg)
        
        logger.info(f'Training samples: {len(train_data)}, validation samples: {len(val_data)}, test samples: {len(test_data)}')
        logger.info(f'Input dimension: {input_dim} (num_sensors={num_sensors} * 2 features)')
    else:
        train_data = prepared_data['train_data']
        val_data = prepared_data['val_data']
        test_data = prepared_data['test_data']
        input_dim = int(prepared_data['input_dim'])
        num_sensors = int(prepared_data['num_sensors'])
        scaler_state = prepared_data.get('label_scaler_state', {'enabled': False})
        def decode_label(x, s=scaler_state):
            if not s.get('enabled', False):
                return x
            method = s.get('method', 'standard')
            if method == 'standard':
                return x * s.get('std', 1.0) + s.get('mean', 0.0)
            elif method == 'minmax':
                return x * s.get('scale', 1.0) + s.get('min', 0.0)
            elif method == 'robust':
                return x * s.get('scale', 1.0) + s.get('median', 0.0)
            return x
        logger.info('[Shared] Using precomputed datasets and splits from prepared_data.')
        logger.info(f'Training samples: {len(train_data)}, validation samples: {len(val_data)}, test samples: {len(test_data)}')
        logger.info(f'Input dimension: {input_dim} (num_sensors={num_sensors} * 2 features)')
    
    logger.info('[5/5] Training model...')
    
    if model_type == 'lstm':
        hidden_dim = int(model_settings.get('hidden_dim', 128))
        num_layers = int(model_settings.get('num_layers', 2))
        dropout_rate = float(model_settings.get('dropout_rate', 0.3))
        bidirectional = bool(model_settings.get('bidirectional', True))
        
        model = LSTMWrapper(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            bidirectional=bidirectional,
            device=device,
        )
    elif model_type == 'gru':
        hidden_dim = int(model_settings.get('hidden_dim', 128))
        num_layers = int(model_settings.get('num_layers', 2))
        dropout_rate = float(model_settings.get('dropout_rate', 0.3))
        bidirectional = bool(model_settings.get('bidirectional', True))
        
        model = GRUWrapper(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            bidirectional=bidirectional,
            device=device,
        )
    elif model_type == 'transformer':
        d_model = int(model_settings.get('d_model', 128))
        nhead = int(model_settings.get('nhead', 4))
        num_layers = int(model_settings.get('num_layers', 3))
        dropout_rate = float(model_settings.get('dropout_rate', 0.2))
        
        model = TransformerWrapper(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            device=device,
        )
    elif model_type == 'cnn_lstm':
        cnn_filters = model_settings.get('cnn_filters', [64, 128])
        cnn_kernel_size = int(model_settings.get('cnn_kernel_size', 3))
        lstm_hidden_dim = int(model_settings.get('lstm_hidden_dim', 128))
        lstm_num_layers = int(model_settings.get('lstm_num_layers', 2))
        dropout_rate = float(model_settings.get('dropout_rate', 0.3))
        
        model = CNNLSTMWrapper(
            input_dim=input_dim,
            cnn_filters=cnn_filters,
            cnn_kernel_size=cnn_kernel_size,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
            dropout_rate=dropout_rate,
            device=device,
        )
    elif model_type == 'tcn':
        num_channels = model_settings.get('num_channels', [64, 128, 256])
        kernel_size = int(model_settings.get('kernel_size', 3))
        dropout_rate = float(model_settings.get('dropout_rate', 0.2))
        
        model = TCNWrapper(
            input_dim=input_dim,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            device=device,
        )
    elif model_type == 'attention_lstm':
        lstm_hidden_dim = int(model_settings.get('lstm_hidden_dim', 128))
        lstm_num_layers = int(model_settings.get('lstm_num_layers', 2))
        attention_dim = int(model_settings.get('attention_dim', 64))
        dropout_rate = float(model_settings.get('dropout_rate', 0.3))
        
        model = AttentionLSTMWrapper(
            input_dim=input_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
            attention_dim=attention_dim,
            dropout_rate=dropout_rate,
            device=device,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    gradient_clipping_config = training_settings.get('gradient_clipping', {})
    warmup_epochs = int(training_settings.get('warmup_epochs', 0))
    
    contrastive_config = training_settings.get('contrastive_learning', {})
    enable_contrastive = contrastive_config.get('enable', False)
    
    if enable_contrastive and model_type in ['lstm', 'gru']:
        print('[4.5/5] Contrastive pre-training...')
        hidden_dim = int(model_settings.get('hidden_dim', 128))
        num_layers = int(model_settings.get('num_layers', 2))
        dropout_rate = float(model_settings.get('dropout_rate', 0.3))
        
        aug_config = contrastive_config.get('augmentation', {})
        augmentation = TimeSeriesAugmentation(
            noise_std=float(aug_config.get('noise_std', 0.01)),
            time_warp_factor=float(aug_config.get('time_warp_factor', 0.1)),
            mask_ratio=float(aug_config.get('mask_ratio', 0.15)),
            enable_noise=bool(aug_config.get('enable_noise', True)),
            enable_time_warp=bool(aug_config.get('enable_time_warp', True)),
            enable_masking=bool(aug_config.get('enable_masking', True)),
        )
        
        bidirectional = bool(model_settings.get('bidirectional', True))
        contrastive_encoder = ContrastiveEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_rate,
            projection_dim=int(contrastive_config.get('projection_dim', 64)),
            encoder_type=model_type,
            bidirectional=bidirectional,
        )
        
        simclr_trainer = SimCLRTrainer(
            encoder=contrastive_encoder,
            augmentation=augmentation,
            device=device,
        )
        
        pretrain_epochs = int(contrastive_config.get('pretrain_epochs', 20))
        pretrain_lr = float(contrastive_config.get('pretrain_lr', 0.001))
        pretrain_batch_size = int(contrastive_config.get('pretrain_batch_size', batch_size))
        
        pretrain_history = simclr_trainer.pretrain(
            train_data=train_data,
            epochs=pretrain_epochs,
            lr=pretrain_lr,
            batch_size=pretrain_batch_size,
            verbose=True,
        )
        
        pretrained_state_dict = simclr_trainer.get_encoder_state_dict()
        
        if model_type == 'lstm':
            model.model.lstm.load_state_dict(pretrained_state_dict)
        elif model_type == 'gru':
            model.model.gru.load_state_dict(pretrained_state_dict)
        
        print('Contrastive pre-training completed. Using pre-trained encoder weights.')
    
    loss_history = model.train(
        train_data,
        val_data,
        epochs=epochs,
        lr=learning_rate,
        batch_size=batch_size,
        verbose=True,
        early_stopping=early_stopping_config,
        scheduler_config=scheduler_config,
        gradient_clipping=gradient_clipping_config,
        warmup_epochs=warmup_epochs,
    )
    
    if train_data.targets is None or val_data.targets is None or test_data.targets is None:
        raise ValueError("Targets are missing in prepared datasets; cannot compute metrics.")
    
    train_pred_encoded = model.predict(train_data, batch_size=batch_size)
    val_pred_encoded = model.predict(val_data, batch_size=batch_size)
    test_pred_encoded = model.predict(test_data, batch_size=batch_size)
    
    train_targets_encoded = train_data.targets
    val_targets_encoded = val_data.targets
    test_targets_encoded = test_data.targets
    
    train_pred = decode_label(train_pred_encoded, scaler_state)
    val_pred = decode_label(val_pred_encoded, scaler_state)
    test_pred = decode_label(test_pred_encoded, scaler_state)
    
    train_targets = decode_label(train_targets_encoded, scaler_state)
    val_targets = decode_label(val_targets_encoded, scaler_state)
    test_targets = decode_label(test_targets_encoded, scaler_state)
    
    rul_transform = dataset_config.get('preprocessing', {}).get('fatigue_life', {}).get('rul_transform')
    if rul_transform == 'log1p':
        def _to_rul(a):
            return np.maximum(0.0, np.expm1(np.asarray(a, dtype=np.float64)))
        train_pred = _to_rul(train_pred)
        val_pred = _to_rul(val_pred)
        test_pred = _to_rul(test_pred)
        train_targets = _to_rul(train_targets)
        val_targets = _to_rul(val_targets)
        test_targets = _to_rul(test_targets)
    
    sample_train_metrics = _regression_metrics(train_targets, train_pred)
    sample_val_metrics = _regression_metrics(val_targets, val_pred)
    sample_test_metrics = _regression_metrics(test_targets, test_pred)
    
    window_train = _aggregate_by_window(train_pred, train_targets, getattr(train_data, "sample_to_window_idx", None))
    window_val = _aggregate_by_window(val_pred, val_targets, getattr(val_data, "sample_to_window_idx", None))
    window_test = _aggregate_by_window(test_pred, test_targets, getattr(test_data, "sample_to_window_idx", None))
    
    window_metrics = {}
    if window_train is not None:
        wt_pred, wt_true = window_train
        wv_pred, wv_true = window_val if window_val is not None else (None, None)
        ws_pred, ws_true = window_test if window_test is not None else (None, None)
        if wt_pred is not None and wv_pred is not None and ws_pred is not None:
            window_metrics.update({
                'train_window': _regression_metrics(wt_true, wt_pred),
                'val_window': _regression_metrics(wv_true, wv_pred),
                'test_window': _regression_metrics(ws_true, ws_pred),
            })
    
    def _format_r2(r2_val):
        return f"{r2_val:.4f}" if not np.isnan(r2_val) else "NaN"
    
    print(f'\n{model_type.upper()} Model Performance (sample-level):')
    print(f"Train RMSE: {sample_train_metrics['rmse']:.4f}, Val RMSE: {sample_val_metrics['rmse']:.4f}, Test RMSE: {sample_test_metrics['rmse']:.4f}")
    print(f"Train MAE: {sample_train_metrics['mae']:.4f}, Val MAE: {sample_val_metrics['mae']:.4f}, Test MAE: {sample_test_metrics['mae']:.4f}")
    print(f"Train R²: {_format_r2(sample_train_metrics['r2'])}, Val R²: {_format_r2(sample_val_metrics['r2'])}, Test R²: {_format_r2(sample_test_metrics['r2'])}")
    print(f"Test R² (RUL>0): {_format_r2(sample_test_metrics['r2_rul_positive'])}")
    
    if window_metrics:
        wtr = window_metrics['train_window']; wvr = window_metrics['val_window']; wsr = window_metrics['test_window']
        print(f'\n{model_type.upper()} Model Performance (window-level):')
        print(f"Train RMSE: {wtr['rmse']:.4f}, Val RMSE: {wvr['rmse']:.4f}, Test RMSE: {wsr['rmse']:.4f}")
        print(f"Train MAE: {wtr['mae']:.4f}, Val MAE: {wvr['mae']:.4f}, Test MAE: {wsr['mae']:.4f}")
        print(f"Train R²: {_format_r2(wtr['r2'])}, Val R²: {_format_r2(wvr['r2'])}, Test R²: {_format_r2(wsr['r2'])}")
        print(f"Test R² (RUL>0, window): {_format_r2(wsr['r2_rul_positive'])}")
    
    metrics = {
        'train_rmse': sample_train_metrics['rmse'],
        'val_rmse': sample_val_metrics['rmse'],
        'test_rmse': sample_test_metrics['rmse'],
        'train_mae': sample_train_metrics['mae'],
        'val_mae': sample_val_metrics['mae'],
        'test_mae': sample_test_metrics['mae'],
        'train_r2': sample_train_metrics['r2'],
        'val_r2': sample_val_metrics['r2'],
        'test_r2': sample_test_metrics['r2'],
        'test_r2_rul_positive': sample_test_metrics['r2_rul_positive'],
        'train_loss_history': loss_history.get('train_loss_history', []),
        'val_loss_history': loss_history.get('val_loss_history', []),
        'epochs': epochs,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
    }
    if window_metrics:
        metrics.update({
            'train_window_rmse': window_metrics['train_window']['rmse'],
            'val_window_rmse': window_metrics['val_window']['rmse'],
            'test_window_rmse': window_metrics['test_window']['rmse'],
            'train_window_mae': window_metrics['train_window']['mae'],
            'val_window_mae': window_metrics['val_window']['mae'],
            'test_window_mae': window_metrics['test_window']['mae'],
            'train_window_r2': window_metrics['train_window']['r2'],
            'val_window_r2': window_metrics['val_window']['r2'],
            'test_window_r2': window_metrics['test_window']['r2'],
            'test_window_r2_rul_positive': window_metrics['test_window']['r2_rul_positive'],
        })
    
    model_filename = f'best_{model_type}_model.pth'
    model_path = os.path.join(model_root, model_filename)
    model.save_model(model_path)
    
    def _convert_nan_to_none(obj):
        if isinstance(obj, dict):
            return {k: _convert_nan_to_none(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_convert_nan_to_none(item) for item in obj]
        elif isinstance(obj, float) and np.isnan(obj):
            return None
        return obj
    
    metrics_filename = f'{model_type}_metrics.json'
    metrics_path = os.path.join(results_dir, metrics_filename)
    with open(metrics_path, 'w', encoding='utf-8') as handle:
        json.dump(_convert_nan_to_none(metrics), handle, indent=2)
    
    print('\n' + '=' * 60)
    print('Training Complete!')
    print('=' * 60)
    print(f'Model saved to: {model_path}')
    print(f'Metrics saved to: {metrics_path}')
    
    return model, metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train deep learning baseline model')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['lstm', 'gru', 'transformer', 'cnn_lstm', 'tcn', 'attention_lstm'],
                       help='Type of model to train')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to raw data file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save model')
    
    args = parser.parse_args()
    
    train_dl_baseline(
        model_type=args.model_type,
        data_path=args.data_path,
        output_dir=args.output_dir,
    )

