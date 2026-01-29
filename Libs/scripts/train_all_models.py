"""
Unified training script for all Deep Learning models.

Usage:
    python -m Libs.scripts.train_all_models --data_path Input/raw/data.mat
"""

import argparse
import json
import logging
import os
import random
import sys
from typing import Any, Dict, Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Libs.config import (load_dataset_config, load_experiment_config,
                         load_model_config)
from Libs.data.dataloader import load_arrows_data
from Libs.data.graph_dataset import GraphDataset
from Libs.data.label_generator import FatigueLabelGenerator
from Libs.data.physics_processor import PhysicsModel
from Libs.data.timeseries_dataset import prepare_timeseries_data
from Libs.scripts.train_dl_baseline import (_build_label_scaler, _split_data,
                                            train_dl_baseline)
from Libs.scripts.train_gnn import train_gnn


def train_all_models(
    data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    prepared_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Trains all enabled Deep Learning models and generates comprehensive evaluation report.
    
    Args:
        data_path: Path to raw data file.
        output_dir: Directory to save models and results.
    
    Returns:
        Dictionary containing training results for all models.
    """
    dataset_config = load_dataset_config()
    model_config = load_model_config()
    experiment_config = load_experiment_config()
    
    logging_cfg = experiment_config.get('logging', {})
    logging.basicConfig(
        level=getattr(logging, logging_cfg.get('level', 'INFO')),
        format=logging_cfg.get('format', '%(asctime)s - %(levelname)s - %(message)s'),
    )
    logger = logging.getLogger(__name__)
    
    def _set_seed(seed: Optional[int]) -> None:
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
    
    def _prepare_shared_data() -> Dict[str, Any]:
        raw_data_path = data_path if data_path is not None else dataset_config.get('data_paths', {}).get('raw_data', 'Input/raw/data.mat')
        
        seed = experiment_config.get('random_state', dataset_config.get('preprocessing', {}).get('data_split', {}).get('random_state', None))
        _set_seed(seed)
        
        data = load_arrows_data(raw_data_path)
        if data is None:
            raise RuntimeError('Failed to load raw dataset.')
        
        physics = PhysicsModel(dataset_config)
        modulus_result = physics.compute_elastic_modulus(data['sensors'], data['time'])
        
        label_generator = FatigueLabelGenerator(dataset_config)
        training_df = label_generator.prepare_training_data(
            modulus_result, data['sensors'], data['time']
        )
        
        cycle_indices = modulus_result.get('cycle_indices', [])
        labels = training_df['rul'].to_numpy()
        valid_mask = ~np.isnan(labels)
        valid_indices = np.where(valid_mask)[0]
        
        time_windows = []
        target_values = []
        for index in valid_indices:
            if index < len(cycle_indices):
                time_windows.append(cycle_indices[index])
                target_values.append(float(labels[index]))
        
        split_settings = dataset_config.get('preprocessing', {}).get('data_split', {})
        test_size = float(split_settings.get('test_size', 0.2))
        validation_size = float(split_settings.get('validation_size', 0.2))
        random_state = split_settings.get('random_state', 42)
        split_method = split_settings.get('split_method', 'temporal')
        ensure_pre_failure_in_train = split_settings.get('ensure_pre_failure_in_train', True)
        min_pre_failure_in_test = split_settings.get('min_pre_failure_in_test')

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
        
        label_norm_cfg = experiment_config.get('preprocessing', {}).get('label_normalization', {})
        encode_label, decode_label, encode_using_state = _build_label_scaler(label_norm_cfg)

        train_labels_encoded, scaler_state = encode_label(np.array(train_labels, dtype=np.float32))
        val_labels_encoded = encode_using_state(np.array(val_labels, dtype=np.float32), scaler_state)
        test_labels_encoded = encode_using_state(np.array(test_labels, dtype=np.float32), scaler_state)
        
        num_sensors = len([k for k in data['sensors'].keys()])
        input_dim = num_sensors * 2
        
        dataset_time_series = dataset_config.get('preprocessing', {}).get('time_series', {})
        exp_time_series = experiment_config.get('preprocessing', {}).get('time_series', {})
        time_series_settings = {**dataset_time_series, **exp_time_series}
        
        max_sequence_length = time_series_settings.get('max_sequence_length', None)
        max_sequence_length = int(max_sequence_length) if max_sequence_length is not None else None
        sequence_length = time_series_settings.get('sequence_length', None)
        sequence_length = int(sequence_length) if sequence_length is not None else None
        stride = time_series_settings.get('stride', None)
        stride = int(stride) if stride is not None else None
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
        
        train_data = prepare_timeseries_data(data['sensors'], train_windows, train_labels_encoded, **common_ts_kwargs)
        val_data = prepare_timeseries_data(data['sensors'], val_windows, val_labels_encoded, **common_ts_kwargs)
        test_data = prepare_timeseries_data(data['sensors'], test_windows, test_labels_encoded, **common_ts_kwargs)
        
        if len(train_data) == 0:
            raise ValueError("No training samples generated; please check preprocessing configuration.")
        
        graph_settings = dataset_config.get('graph_dataset', {})
        graph_type = graph_settings.get('graph_type', 'stress_propagation')
        preferred_device = graph_settings.get('device', 'cpu')
        graph_device = preferred_device
        
        available_sensor_positions = physics.get_available_sensor_positions(data['sensors'])
        num_nodes = len(available_sensor_positions)
        graph_dataset = GraphDataset(available_sensor_positions, graph_type=graph_type, device=graph_device)
        train_graph_data = graph_dataset.prepare_graph_data(data['sensors'], train_windows, train_labels)
        val_graph_data = graph_dataset.prepare_graph_data(data['sensors'], val_windows, val_labels)
        test_graph_data = graph_dataset.prepare_graph_data(data['sensors'], test_windows, test_labels)
        
        return {
            'dl_data': {
                'train_data': train_data,
                'val_data': val_data,
                'test_data': test_data,
                'input_dim': input_dim,
                'num_sensors': num_sensors,
                'label_scaler_state': scaler_state,
            },
            'graph_data': {
                'train_graph_data': train_graph_data,
                'val_graph_data': val_graph_data,
                'test_graph_data': test_graph_data,
                'train_labels': train_labels,
                'val_labels': val_labels,
                'test_labels': test_labels,
                'num_nodes': num_nodes,
            }
        }
    
    model_paths = model_config.get('model_paths', {})
    output_root = output_dir if output_dir is not None else model_paths.get('models_dir', 'Output/models')
    results_dir = model_paths.get('results_dir', 'Output/results')
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print('=' * 80)
    print('Training All Deep Learning Models')
    print('=' * 80)
    
    all_results = {}
    
    raw_data_path = data_path if data_path is not None else dataset_config.get('data_paths', {}).get('raw_data', 'Input/raw/data.mat')
    
    dl_models_config = model_config.get('deep_learning_models', {})
    shared = prepared_data if prepared_data is not None else _prepare_shared_data()
    
    for model_name, model_cfg in dl_models_config.items():
        if not model_cfg.get('enable', False):
            continue
        
        logger.info('=' * 80)
        logger.info(f'Training {model_name.upper()} Model')
        logger.info('=' * 80)
        
        if model_name == 'gnn':
            model, metrics = train_gnn(
                data_path=raw_data_path,
                output_dir=output_root,
                dataset_config=dataset_config,
                model_config=model_config,
                experiment_config=experiment_config,
                prepared_data=shared.get('graph_data'),
            )
        else:
            model, metrics = train_dl_baseline(
                model_type=model_name,
                data_path=raw_data_path,
                output_dir=output_root,
                dataset_config=dataset_config,
                model_config=model_config,
                experiment_config=experiment_config,
                prepared_data=shared.get('dl_data'),
            )
        
        all_results[model_name] = metrics
    
    print('\n' + '=' * 80)
    print('Training Summary - All Models')
    print('=' * 80)
    
    for model_name, metrics in all_results.items():
        print(f'\n{model_name.upper()} Model Results:')
        print(f"  Train RMSE: {metrics.get('train_rmse', 'N/A'):.4f}")
        print(f"  Val RMSE: {metrics.get('val_rmse', 'N/A'):.4f}")
        print(f"  Test RMSE: {metrics.get('test_rmse', 'N/A'):.4f}")
        print(f"  Train MAE: {metrics.get('train_mae', 'N/A'):.4f}")
        print(f"  Val MAE: {metrics.get('val_mae', 'N/A'):.4f}")
        print(f"  Test MAE: {metrics.get('test_mae', 'N/A'):.4f}")
        print(f"  Train R²: {metrics.get('train_r2', 'N/A'):.4f}")
        print(f"  Val R²: {metrics.get('val_r2', 'N/A'):.4f}")
        print(f"  Test R²: {metrics.get('test_r2', 'N/A'):.4f}")
        r2_pos = metrics.get('test_r2_rul_positive')
        print(f"  Test R² (RUL>0): {r2_pos:.4f}" if r2_pos is not None and not (isinstance(r2_pos, float) and np.isnan(r2_pos)) else "  Test R² (RUL>0): N/A")
    
    def _nan_to_none(obj):
        if isinstance(obj, dict):
            return {k: _nan_to_none(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_nan_to_none(x) for x in obj]
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    results_path = os.path.join(results_dir, 'all_dl_models_results.json')
    with open(results_path, 'w', encoding='utf-8') as handle:
        json.dump(_nan_to_none(all_results), handle, indent=2, default=str)
    
    print(f'\nResults saved to: {results_path}')
    print('=' * 80)
    
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train all enabled Deep Learning models')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to raw data file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save models and results')
    
    args = parser.parse_args()
    
    train_all_models(
        data_path=args.data_path,
        output_dir=args.output_dir,
    )

