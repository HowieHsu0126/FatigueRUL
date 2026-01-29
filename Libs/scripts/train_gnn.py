"""Configuration-driven training script for the spatio-temporal GNN."""

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Libs.config import (load_dataset_config, load_experiment_config,
                         load_model_config)
from Libs.data.dataloader import load_arrows_data
from Libs.data.graph_dataset import GraphDataset
from Libs.data.label_generator import FatigueLabelGenerator
from Libs.data.physics_processor import PhysicsModel
from Libs.models.networks.spatiotemporal_gnn import SpatioTemporalGNN
from Libs.scripts.train_dl_baseline import _build_sample_weights


def _resolve_device(preferred: str) -> str:
    """Returns a runtime-safe compute device string based on availability.

    Args:
        preferred: Target device identifier sourced from configuration.

    Returns:
        Device string guaranteed to be usable in the current environment.
    """
    if preferred == 'cuda' and not torch.cuda.is_available():
        return 'cpu'
    return preferred


def _build_label_scaler(config: Dict[str, Any]):
    """Returns (encode_fn, decode_fn, encode_using_state_fn).
    
    encode_fn(arr) -> (encoded, state). decode_fn(arr, state) -> decoded.
    encode_using_state_fn(arr, state) -> encoded; use for val/test with train state.
    
    Args:
        config: Label normalization configuration dictionary.
        
    Returns:
        Tuple of (encode function, decode function, encode_using_state function).
    """
    enable = config.get('enable', False)
    method = config.get('method', 'standard')
    
    if not enable:
        def _encode_noop(x):
            return (x, {'enabled': False})
        def _decode_noop(x, s):
            return x
        def _encode_using_state_noop(arr, state):
            return np.asarray(arr, dtype=np.float32) if not isinstance(arr, np.ndarray) else arr
        return _encode_noop, _decode_noop, _encode_using_state_noop

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
        def _encode_noop(x):
            return (x, {'enabled': False})
        def _decode_noop(x, s):
            return x
        def _encode_using_state_noop(arr, state):
            return np.asarray(arr, dtype=np.float32) if not isinstance(arr, np.ndarray) else arr
        return _encode_noop, _decode_noop, _encode_using_state_noop

    def encode_with_state(arr: np.ndarray):
        encoded, state = encode(arr)
        return encoded, state

    return encode_with_state, decode, encode_using_state


def _split_windows(
    time_windows: Sequence[Tuple[int, int]],
    labels: Sequence[float],
    test_size: float,
    validation_size: float,
    random_state: Optional[int],
    split_method: str = "temporal",
    ensure_pre_failure_in_train: bool = True,
    min_pre_failure_in_test: Optional[int] = None,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]], np.ndarray, np.ndarray, np.ndarray]:
    """Partitions graph samples into train, validation, and test splits with balanced pre-failure/post-failure distribution.

    Args:
        time_windows: Sliding window indices for each sample.
        labels: Target values associated with each window.
        test_size: Fraction of samples reserved for testing.
        validation_size: Fraction of samples reserved for validation (from training set).
        random_state: Seed controlling deterministic splits.
        split_method: "temporal" for time-series split or "random" for random split.
        ensure_pre_failure_in_train: If True, ensures training set contains pre-failure data.
        min_pre_failure_in_test: If set, ensure test has at least this many pre-failure samples (temporal only).

    Returns:
        Tuple of (train_windows, val_windows, test_windows, train_labels, val_labels, test_labels).
    """
    indices = np.arange(len(time_windows))
    labels_array = np.asarray(labels)

    if split_method == "temporal":
        pre_failure_mask = labels_array > 0
        post_failure_mask = labels_array == 0
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
        
        pre_in_train = np.sum(labels_array[train_idx] > 0)
        pre_in_val = np.sum(labels_array[val_idx] > 0)
        pre_in_test = np.sum(labels_array[test_idx] > 0)
        
        print(f"[Info] _split_windows: Balanced split - Train: {len(train_idx)} samples ({pre_in_train} pre-failure, {len(train_idx)-pre_in_train} post-failure), "
              f"Val: {len(val_idx)} samples ({pre_in_val} pre-failure, {len(val_idx)-pre_in_val} post-failure), "
              f"Test: {len(test_idx)} samples ({pre_in_test} pre-failure, {len(test_idx)-pre_in_test} post-failure)")
    else:
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)
        
        val_size_from_train = validation_size / (1.0 - test_size)
        train_idx, val_idx = train_test_split(train_idx, test_size=val_size_from_train, random_state=random_state)

    train_windows = [time_windows[i] for i in train_idx]
    val_windows = [time_windows[i] for i in val_idx]
    test_windows = [time_windows[i] for i in test_idx]
    train_labels = np.asarray([labels[i] for i in train_idx], dtype=np.float32)
    val_labels = np.asarray([labels[i] for i in val_idx], dtype=np.float32)
    test_labels = np.asarray([labels[i] for i in test_idx], dtype=np.float32)

    return train_windows, val_windows, test_windows, train_labels, val_labels, test_labels


def train_gnn(
    data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    dataset_config: Optional[Dict[str, Any]] = None,
    model_config: Optional[Dict[str, Any]] = None,
    experiment_config: Optional[Dict[str, Any]] = None,
    prepared_data: Optional[Dict[str, Any]] = None,
) -> Tuple[SpatioTemporalGNN, Dict[str, Any]]:
    """Trains the spatio-temporal GNN using project configuration files.

    Args:
        data_path: Optional override for the raw dataset file path.
        output_dir: Optional override for the model output directory.
        dataset_config: Optional dataset configuration dictionary.
        model_config: Optional model configuration dictionary.
        experiment_config: Optional experiment configuration dictionary.

    Returns:
        Tuple with the trained model wrapper and aggregated training metrics.
    """
    dataset_cfg = load_dataset_config() if dataset_config is None else dataset_config
    model_cfg = load_model_config() if model_config is None else model_config
    experiment_cfg = load_experiment_config() if experiment_config is None else experiment_config

    data_paths = dict(dataset_cfg.get('data_paths', {}))
    graph_settings = dict(dataset_cfg.get('graph_dataset', {}))
    split_settings = dict(dataset_cfg.get('preprocessing', {}).get('data_split', {}))
    training_settings = dict(experiment_cfg.get('training', {}))

    model_paths = dict(model_cfg.get('model_paths', {}))
    gnn_settings = dict(model_cfg.get('deep_learning_models', {}).get('gnn', {}))

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

    epochs = int(training_settings.get('num_epochs', 100))
    learning_rate = float(training_settings.get('learning_rate', 0.001))
    batch_size = int(training_settings.get('batch_size', 32))
    
    early_stopping_config = training_settings.get('early_stopping', {})
    early_stopping_config['enable'] = early_stopping_config.get('enable', True)
    scheduler_config = training_settings.get('scheduler', {})
    if scheduler_config.get('type') == 'reduce_on_plateau':
        reduce_config = scheduler_config.get('reduce_on_plateau', {})
        scheduler_config['factor'] = reduce_config.get('factor', 0.5)
        scheduler_config['patience'] = reduce_config.get('patience', 10)

    graph_type = graph_settings.get('graph_type', 'stress_propagation')

    preferred_device = gnn_settings.get('device', 'cuda')
    device = _resolve_device(preferred_device)
    graph_device = device
    hidden_dim = int(gnn_settings.get('hidden_dim', 64))
    num_layers = int(gnn_settings.get('num_layers', 2))
    # Must match node feature count from GraphDataset.prepare_graph_data (force/disp: mean, std, max, min, slope).
    input_dim = int(gnn_settings.get('input_dim', 10))

    label_norm_cfg = experiment_cfg.get('preprocessing', {}).get('label_normalization', {})
    encode_label, decode_label, encode_using_state = _build_label_scaler(label_norm_cfg)
    loss_config = training_settings.get('loss_function', {})

    logger = logging.getLogger(__name__)
    
    logger.info('=' * 60)
    logger.info('Training Spatio-Temporal GNN Model')
    logger.info('=' * 60)
    
    if prepared_data is None:
        logger.info('[1/5] Loading ARROWS data...')
        data = load_arrows_data(raw_data_path)
        if data is None:
            raise RuntimeError('Failed to load raw dataset.')
        
        logger.info('[2/5] Computing elastic modulus...')
        physics = PhysicsModel(dataset_cfg)
        modulus_result = physics.compute_elastic_modulus(data['sensors'], data['time'])
        
        logger.info('[3/5] Building training dataframe...')
        label_generator = FatigueLabelGenerator(dataset_cfg)
        training_df = label_generator.prepare_training_data(modulus_result, data['sensors'], data['time'])
        
        available_sensor_positions = physics.get_available_sensor_positions(data['sensors'])
        num_nodes = len(available_sensor_positions)
        graph_dataset = GraphDataset(available_sensor_positions, graph_type=graph_type, device=graph_device)
        
        cycle_indices = modulus_result.get('cycle_indices', [])
        labels_array = training_df['rul'].to_numpy()
        valid_mask = ~np.isnan(labels_array)
        valid_indices = np.where(valid_mask)[0]
        
        all_windows: List[Tuple[int, int]] = []
        all_labels: List[float] = []
        for index in valid_indices:
            if index < len(cycle_indices):
                all_windows.append(cycle_indices[index])
                all_labels.append(float(labels_array[index]))
        
        if len(all_windows) == 0:
            raise RuntimeError('No valid samples available for graph dataset.')
        
        min_pre_failure_in_test = split_settings.get('min_pre_failure_in_test')
        train_windows, val_windows, test_windows, train_labels, val_labels, test_labels = _split_windows(
            all_windows,
            all_labels,
            test_size,
            validation_size,
            random_state,
            split_method=split_method,
            ensure_pre_failure_in_train=ensure_pre_failure_in_train,
            min_pre_failure_in_test=min_pre_failure_in_test,
        )
        train_labels = np.array(train_labels, dtype=np.float32)
        
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
        
        train_sample_weights = _build_sample_weights(np.asarray(train_labels, dtype=np.float32), loss_config.get('sample_weighting', {}))
        train_labels_encoded, scaler_state = encode_label(np.array(train_labels, dtype=np.float32))
        val_labels_encoded = encode_using_state(np.array(val_labels, dtype=np.float32), scaler_state)
        test_labels_encoded = encode_using_state(np.array(test_labels, dtype=np.float32), scaler_state)
        
        logger.info('[4/5] Preparing graph samples...')
        train_graph_data = graph_dataset.prepare_graph_data(data['sensors'], train_windows, train_labels_encoded)
        val_graph_data = graph_dataset.prepare_graph_data(data['sensors'], val_windows, val_labels_encoded)
        test_graph_data = graph_dataset.prepare_graph_data(data['sensors'], test_windows, test_labels_encoded)
    else:
        train_graph_data = prepared_data['train_graph_data']
        val_graph_data = prepared_data['val_graph_data']
        test_graph_data = prepared_data['test_graph_data']
        train_labels = prepared_data['train_labels']
        val_labels = prepared_data['val_labels']
        test_labels = prepared_data['test_labels']
        num_nodes = prepared_data.get('num_nodes', 0)
        logger.info('[Shared] Using precomputed graph datasets and splits from prepared_data.')
        train_sample_weights = _build_sample_weights(np.asarray(train_labels, dtype=np.float32), loss_config.get('sample_weighting', {}))
        
        train_labels_encoded, scaler_state = encode_label(np.array(train_labels, dtype=np.float32))
        val_labels_encoded = encode_using_state(np.array(val_labels, dtype=np.float32), scaler_state)
        test_labels_encoded = encode_using_state(np.array(test_labels, dtype=np.float32), scaler_state)
        
        for idx, data in enumerate(train_graph_data):
            data.y = torch.tensor([[train_labels_encoded[idx]]], dtype=torch.float32, device=data.x.device)
        for idx, data in enumerate(val_graph_data):
            data.y = torch.tensor([[val_labels_encoded[idx]]], dtype=torch.float32, device=data.x.device)
        for idx, data in enumerate(test_graph_data):
            data.y = torch.tensor([[test_labels_encoded[idx]]], dtype=torch.float32, device=data.x.device)

    # Normalize node features using training set statistics to stabilize GNN training
    all_train_x = torch.cat([g.x for g in train_graph_data], dim=0)
    feat_mean = all_train_x.mean(dim=0)
    feat_std = all_train_x.std(dim=0) + 1e-8
    for g in train_graph_data + val_graph_data + test_graph_data:
        g.x = (g.x - feat_mean.to(g.x.device)) / feat_std.to(g.x.device)

    gnn = SpatioTemporalGNN(
        num_nodes=num_nodes,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        device=device,
    )

    logger.info(f'Training samples: {len(train_graph_data)}, Validation samples: {len(val_graph_data)}, Test samples: {len(test_graph_data)}')
    logger.info('[5/5] Training GNN model...')
    gradient_clipping_config = training_settings.get('gradient_clipping', {})
    warmup_epochs = int(training_settings.get('warmup_epochs', 0))
    
    loss_history_dict = gnn.train(
        train_graph_data,
        epochs=epochs,
        lr=learning_rate,
        batch_size=batch_size,
        verbose=True,
        val_data=val_graph_data,
        early_stopping=early_stopping_config,
        scheduler_config=scheduler_config,
        gradient_clipping=gradient_clipping_config,
        warmup_epochs=warmup_epochs,
        sample_weights=train_sample_weights,
    )

    train_pred_encoded = gnn.predict(train_graph_data)
    val_pred_encoded = gnn.predict(val_graph_data)
    test_pred_encoded = gnn.predict(test_graph_data)

    train_pred = decode_label(np.asarray(train_pred_encoded), scaler_state)
    val_pred = decode_label(np.asarray(val_pred_encoded), scaler_state)
    test_pred = decode_label(np.asarray(test_pred_encoded), scaler_state)

    rul_transform = dataset_cfg.get('preprocessing', {}).get('fatigue_life', {}).get('rul_transform')
    if rul_transform == 'log1p':
        def _to_rul(a):
            return np.maximum(0.0, np.expm1(np.asarray(a, dtype=np.float64)))
        train_pred = _to_rul(train_pred)
        val_pred = _to_rul(val_pred)
        test_pred = _to_rul(test_pred)
        train_labels = _to_rul(train_labels)
        val_labels = _to_rul(val_labels)
        test_labels = _to_rul(test_labels)

    train_rmse = float(np.sqrt(mean_squared_error(train_labels, train_pred)))
    val_rmse = float(np.sqrt(mean_squared_error(val_labels, val_pred)))
    test_rmse = float(np.sqrt(mean_squared_error(test_labels, test_pred)))
    train_mae = float(mean_absolute_error(train_labels, train_pred))
    val_mae = float(mean_absolute_error(val_labels, val_pred))
    test_mae = float(mean_absolute_error(test_labels, test_pred))
    
    def _compute_r2(y_true, y_pred):
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = np.nan if ss_tot == 0 else float(r2_score(y_true, y_pred))
        if ss_tot == 0:
            logger.warning(f"R² cannot be computed (SS_tot=0, all labels are identical: {np.mean(y_true):.4f})")
        return r2
    
    train_r2 = _compute_r2(train_labels, train_pred)
    val_r2 = _compute_r2(val_labels, val_pred)
    test_r2 = _compute_r2(test_labels, test_pred)
    
    mask_pos = test_labels > 0
    n_pos = int(np.sum(mask_pos))
    if n_pos >= 2:
        ss_pos = np.sum((test_labels[mask_pos] - np.mean(test_labels[mask_pos])) ** 2)
        test_r2_rul_positive = np.nan if ss_pos == 0 else float(r2_score(test_labels[mask_pos], test_pred[mask_pos]))
    else:
        test_r2_rul_positive = np.nan
    
    def _format_r2(r2_val):
        return f"{r2_val:.4f}" if not np.isnan(r2_val) else "NaN"

    logger.info('GNN Model Performance:')
    logger.info(f'Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}, Test RMSE: {test_rmse:.4f}')
    logger.info(f'Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}, Test MAE: {test_mae:.4f}')
    logger.info(f'Train R²: {_format_r2(train_r2)}, Val R²: {_format_r2(val_r2)}, Test R²: {_format_r2(test_r2)}')
    logger.info(f'Test R² (RUL>0): {_format_r2(test_r2_rul_positive)}')

    metrics = {
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'val_mae': val_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'val_r2': val_r2,
        'test_r2': test_r2,
        'test_r2_rul_positive': test_r2_rul_positive,
        'train_loss_history': loss_history_dict.get('train_loss', []),
        'val_loss_history': loss_history_dict.get('val_loss', []),
        'epochs': epochs,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
    }

    model_path = model_paths.get('best_model', os.path.join(model_root, 'best_model.pth'))
    gnn.save_model(model_path)

    def _convert_nan_to_none(obj):
        if isinstance(obj, dict):
            return {k: _convert_nan_to_none(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_convert_nan_to_none(item) for item in obj]
        elif isinstance(obj, float) and np.isnan(obj):
            return None
        return obj
    
    metrics_path = os.path.join(results_dir, 'gnn_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as handle:
        json.dump(_convert_nan_to_none(metrics), handle, indent=2)

    logger.info('=' * 60)
    logger.info('Training Complete!')
    logger.info('=' * 60)
    logger.info(f'Saved model: {model_path}')
    logger.info(f'Saved metrics: {metrics_path}')

    return gnn, metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the spatio-temporal GNN using project configuration files.')
    parser.add_argument('--data_path', type=str, default=None, help='Optional override for raw data file path.')
    parser.add_argument('--output_dir', type=str, default=None, help='Optional override for model output directory.')

    arguments = parser.parse_args()

    train_gnn(arguments.data_path, arguments.output_dir)

