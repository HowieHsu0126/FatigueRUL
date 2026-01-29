"""
Script to prepare and build the dataset from raw ARROWS data.

This script:
1. Loads raw .mat data
2. Computes elastic modulus using physics model
3. Generates fatigue life labels
4. Saves processed dataset
"""

import os
import sys
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Libs.config import load_dataset_config
from Libs.data.dataloader import load_arrows_data
from Libs.data.label_generator import FatigueLabelGenerator
from Libs.data.physics_processor import PhysicsModel


def prepare_dataset(
    data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    dataset_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Builds processed datasets according to configuration settings."""
    config = load_dataset_config() if dataset_config is None else dataset_config
    data_paths = dict(config.get('data_paths', {}))

    if data_path is not None:
        data_paths['raw_data'] = data_path

    processed_root = output_dir if output_dir is not None else data_paths.get('processed_data_dir', 'Input/processed')
    data_paths['processed_data_dir'] = processed_root

    file_keys = ['processed_training_data', 'physics_results', 'processed_metadata']
    for key in file_keys:
        if key in data_paths:
            filename = os.path.basename(data_paths[key])
            data_paths[key] = os.path.join(processed_root, filename)

    os.makedirs(processed_root, exist_ok=True)
    for key in file_keys:
        if key in data_paths:
            os.makedirs(os.path.dirname(data_paths[key]), exist_ok=True)
    
    print("=" * 60)
    print("Dataset Preparation Pipeline")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n[1/4] Loading ARROWS data...")
    data = load_arrows_data(data_paths['raw_data'])
    
    if data is None:
        print("Error: Failed to load data.")
        return None
    
    # Step 2: Compute elastic modulus
    print("\n[2/4] Computing elastic modulus...")
    physics = PhysicsModel(config)
    modulus_result = physics.compute_elastic_modulus(data['sensors'], data['time'])
    
    # Step 3: Generate labels
    print("\n[3/4] Generating fatigue life labels...")
    label_gen = FatigueLabelGenerator(dataset_config)
    training_df = label_gen.prepare_training_data(
        modulus_result,
        data['sensors'],
        data['time']
    )
    
    # Step 4: Save processed data
    print("\n[4/4] Saving processed dataset...")
    
    # Save training dataframe
    training_df_path = data_paths.get('processed_training_data', os.path.join(processed_root, 'training_dataset.csv'))
    training_df.to_csv(training_df_path, index=False)
    print(f"  - Training dataset saved to {training_df_path}")
    
    # Save physics results
    physics_result_path = data_paths.get('physics_results', os.path.join(processed_root, 'physics_results.pkl'))
    joblib.dump(modulus_result, physics_result_path)
    print(f"  - Physics results saved to {physics_result_path}")
    
    # Save metadata
    metadata = {
        'num_samples': len(training_df),
        'num_features': len(training_df.columns) - 1,  # Exclude label
        'fatigue_life': training_df['fatigue_life'].iloc[0] if len(training_df) > 0 else None,
        'initial_modulus': label_gen.compute_initial_modulus(modulus_result['elastic_modulus']),
    }
    
    metadata_path = data_paths.get('processed_metadata', os.path.join(processed_root, 'dataset_metadata.pkl'))
    joblib.dump(metadata, metadata_path)
    print(f"  - Metadata saved to {metadata_path}")
    
    print("\n" + "=" * 60)
    print("Dataset Preparation Complete!")
    print("=" * 60)
    print(f"\nDataset Summary:")
    fatigue_life = metadata['fatigue_life']
    fatigue_life_msg = f"{fatigue_life:.0f} cycles" if fatigue_life is not None else "N/A"
    initial_modulus_msg = f"{metadata['initial_modulus']:.2f}" if metadata['initial_modulus'] is not None else "N/A"
    print(f"  - Samples: {len(training_df)}")
    print(f"  - Features: {len(training_df.columns) - 1}")
    print(f"  - Fatigue Life: {fatigue_life_msg}")
    print(f"  - Initial Modulus: {initial_modulus_msg}")
    
    return {
        'training_df': training_df,
        'modulus_result': modulus_result,
        'metadata': metadata,
        'paths': data_paths,
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare dataset from raw ARROWS data')
    parser.add_argument('--data_path', type=str, default='Input/raw/data.mat',
                       help='Path to raw .mat data file')
    parser.add_argument('--output_dir', type=str, default='Output',
                       help='Directory to save processed data')
    
    args = parser.parse_args()
    
    prepare_dataset(args.data_path, args.output_dir)

