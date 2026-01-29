"""Graph data preparation script driven by configuration files."""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Libs.config import load_dataset_config
from Libs.data.dataloader import load_arrows_data
from Libs.data.graph_dataset import GraphDataset
from Libs.data.label_generator import FatigueLabelGenerator
from Libs.data.physics_processor import PhysicsModel


def _resolve_graph_paths(data_paths: Dict[str, Any], graph_dir: str) -> Tuple[str, str]:
    """Builds absolute paths for graph tensors and metadata files.

    Args:
        data_paths: Mapping of dataset path keys to filesystem locations.
        graph_dir: Directory where graph artifacts should be written.

    Returns:
        Tuple containing the data tensor path and metadata path.
    """
    graph_file = data_paths.get('graph_data_file')
    metadata_file = data_paths.get('graph_metadata')

    if graph_file is None:
        graph_file = os.path.join(graph_dir, 'graph_data.pt')
    if metadata_file is None:
        metadata_file = os.path.join(graph_dir, 'graph_metadata.json')

    return graph_file, metadata_file


def prepare_graph_data(
    data_path: Optional[str] = None,
    dataset_config: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Any], GraphDataset]:
    """Constructs graph-structured datasets according to configuration settings.

    Args:
        data_path: Optional override for the raw dataset file path.
        dataset_config: Optional dictionary supplying dataset configuration values.

    Returns:
        Tuple with the generated graph samples and the dataset helper instance.
    """
    config = load_dataset_config() if dataset_config is None else dataset_config
    data_paths = dict(config.get('data_paths', {}))
    graph_settings = dict(config.get('graph_dataset', {}))

    raw_data_path = data_path if data_path is not None else data_paths.get('raw_data', 'Input/raw/data.mat')
    processed_root = data_paths.get('processed_data_dir', 'Input/processed')
    graph_root = data_paths.get('graph_processed_dir', 'Input/graph_processed')

    os.makedirs(processed_root, exist_ok=True)
    os.makedirs(graph_root, exist_ok=True)

    graph_data_path, metadata_path = _resolve_graph_paths(data_paths, graph_root)
    os.makedirs(os.path.dirname(graph_data_path), exist_ok=True)
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

    print('=' * 60)
    print('Graph Data Preparation Pipeline')
    print('=' * 60)

    print('\n[1/5] Loading ARROWS data...')
    data = load_arrows_data(raw_data_path)
    if data is None:
        raise RuntimeError('Failed to load raw dataset.')

    print('\n[2/5] Computing elastic modulus...')
    physics = PhysicsModel(config)
    modulus_result = physics.compute_elastic_modulus(data['sensors'], data['time'])

    print('\n[3/5] Generating fatigue life labels...')
    label_generator = FatigueLabelGenerator(config)
    training_df = label_generator.prepare_training_data(modulus_result, data['sensors'], data['time'])

    graph_type = graph_settings.get('graph_type', 'stress_propagation')
    device = graph_settings.get('device', 'cpu')

    print(f"\n[4/5] Preparing graph data (type: {graph_type})...")
    available_sensor_positions = physics.get_available_sensor_positions(data['sensors'])
    graph_dataset = GraphDataset(available_sensor_positions, graph_type=graph_type, device=device)

    cycle_indices = modulus_result.get('cycle_indices', [])
    labels = training_df['rul'].to_numpy()

    valid_mask = ~np.isnan(labels)
    valid_indices = np.where(valid_mask)[0]

    time_windows: List[Tuple[int, int]] = []
    target_values: List[float] = []
    for index in valid_indices:
        if index < len(cycle_indices):
            time_windows.append(cycle_indices[index])
            target_values.append(float(labels[index]))

    graph_data = graph_dataset.prepare_graph_data(data['sensors'], time_windows, np.array(target_values))

    metadata = {
        'graph_type': graph_type,
        'num_samples': len(graph_data),
        'time_windows': [[int(w[0]), int(w[1])] for w in time_windows],
        'labels': [float(v) for v in target_values],
    }

    print(f"\n[5/5] Saving graph data to {graph_data_path}...")
    graph_dataset.save_graph_data(
        graph_data,
        output_dir=os.path.dirname(graph_data_path),
        filename=os.path.basename(graph_data_path),
        metadata=metadata,
    )

    with open(metadata_path, 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2)

    print('\n' + '=' * 60)
    print('Graph Data Preparation Complete!')
    print('=' * 60)
    print('\nSummary:')
    print(f"  - Graph type: {graph_type}")
    print(f"  - Number of samples: {len(graph_data)}")
    print(f"  - Number of nodes: {graph_dataset.num_nodes}")
    print(f"  - Number of edges: {graph_dataset.edge_index.shape[1]}")
    print(f"  - Saved graphs: {graph_data_path}")
    print(f"  - Metadata: {metadata_path}")

    return graph_data, graph_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare and save graph data using configuration files.')
    parser.add_argument('--data_path', type=str, default=None, help='Optional override for raw data file path.')

    arguments = parser.parse_args()

    prepare_graph_data(arguments.data_path)

