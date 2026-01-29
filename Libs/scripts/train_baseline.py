"""
Script to train baseline model (LightGBM/XGBoost).

Usage:
    python -m Libs.scripts.train_baseline --data_path Output/training_dataset.csv
"""

import os
import sys
import argparse
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Libs.models.networks.baselines.baseline import BaselineModel


def train_baseline(data_path='Output/training_dataset.csv', 
                  model_type='lightgbm',
                  output_dir='Output/models',
                  test_size=0.2,
                  validation_size=0.2,
                  dataset_config=None):
    """
    Trains baseline model and saves it.
    
    Args:
        data_path: Path to training dataset CSV.
        model_type: Type of model ('lightgbm' or 'xgboost').
        output_dir: Directory to save trained model.
        test_size: Fraction of data for testing.
        validation_size: Fraction of data for validation (from training set).
        dataset_config: Optional dataset configuration dictionary.
    """
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from Libs.config import load_dataset_config
    
    if dataset_config is None:
        dataset_config = load_dataset_config()
    
    split_settings = dataset_config.get('preprocessing', {}).get('data_split', {})
    if test_size is None:
        test_size = float(split_settings.get('test_size', 0.2))
    else:
        test_size = float(test_size)
    if validation_size is None:
        validation_size = float(split_settings.get('validation_size', 0.2))
    else:
        validation_size = float(validation_size)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"Training Baseline Model ({model_type.upper()})")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading dataset from {data_path}...")
    training_df = pd.read_csv(data_path)
    print(f"Loaded {len(training_df)} samples")
    
    # Train model
    print("\nTraining model...")
    baseline = BaselineModel(model_type=model_type)
    results = baseline.train(training_df, test_size=test_size, validation_size=validation_size)
    
    # Save model
    model_path = os.path.join(output_dir, f'baseline_{model_type}.pkl')
    baseline.save_model(model_path)
    
    # Save feature importance
    importance_df = baseline.get_feature_importance()
    importance_path = os.path.join(output_dir, f'baseline_{model_type}_importance.csv')
    importance_df.to_csv(importance_path, index=False)
    print(f"\nFeature importance saved to {importance_path}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    return baseline, results


if __name__ == '__main__':
    from Libs.config import load_dataset_config
    
    parser = argparse.ArgumentParser(description='Train baseline model')
    parser.add_argument('--data_path', type=str, default='Input/processed/training_dataset.csv',
                       help='Path to training dataset CSV')
    parser.add_argument('--model_type', type=str, default='lightgbm',
                       choices=['lightgbm', 'xgboost'],
                       help='Type of baseline model')
    parser.add_argument('--output_dir', type=str, default='Output/models',
                       help='Directory to save trained model')
    parser.add_argument('--test_size', type=float, default=None,
                       help='Fraction of data for testing (overrides config)')
    parser.add_argument('--validation_size', type=float, default=None,
                       help='Fraction of data for validation (overrides config)')
    
    args = parser.parse_args()
    
    dataset_config = load_dataset_config()
    
    train_baseline(
        args.data_path,
        args.model_type,
        args.output_dir,
        args.test_size if args.test_size is not None else None,
        args.validation_size if args.validation_size is not None else None,
        dataset_config
    )

