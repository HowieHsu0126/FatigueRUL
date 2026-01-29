"""
Full pipeline script: prepare dataset, train models, and evaluate.

This script runs the complete pipeline:
1. Prepare dataset
2. Train baseline model
3. Train GNN model
4. Evaluate and compare models
5. Generate interpretability analysis
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Libs.config import load_dataset_config
from Libs.data.physics_processor import PhysicsModel
from Libs.scripts.prepare_dataset import prepare_dataset
from Libs.scripts.train_baseline import train_baseline
from Libs.scripts.train_gnn import train_gnn
from Libs.exps.evaluator import ModelEvaluator
from Libs.utils.interpretability import ModelInterpretability
import numpy as np


def run_full_pipeline(data_path='Input/raw/data.mat', output_dir='Output'):
    """
    Runs the complete pipeline.
    
    Args:
        data_path: Path to raw .mat data file.
        output_dir: Directory to save all outputs.
    """
    print("=" * 80)
    print("ARROWS Fatigue Life Prediction - Full Pipeline")
    print("=" * 80)
    
    # Step 1: Prepare dataset
    print("\n" + "=" * 80)
    print("STEP 1: Dataset Preparation")
    print("=" * 80)
    dataset = prepare_dataset(data_path, output_dir)
    
    if dataset is None:
        print("Error: Dataset preparation failed. Exiting.")
        return
    
    # Step 2: Train baseline model
    print("\n" + "=" * 80)
    print("STEP 2: Baseline Model Training")
    print("=" * 80)
    training_df_path = os.path.join(output_dir, 'training_dataset.csv')
    baseline, baseline_results = train_baseline(
        training_df_path,
        model_type='lightgbm',
        output_dir=os.path.join(output_dir, 'models')
    )
    
    # Step 3: Train GNN model
    print("\n" + "=" * 80)
    print("STEP 3: GNN Model Training")
    print("=" * 80)
    gnn, gnn_results = train_gnn(
        data_path,
        output_dir=os.path.join(output_dir, 'models'),
        epochs=50
    )
    
    # Step 4: Evaluate and compare
    print("\n" + "=" * 80)
    print("STEP 4: Model Evaluation and Comparison")
    print("=" * 80)
    evaluator = ModelEvaluator(output_dir=output_dir)
    prepared_data = evaluator.prepare_data(data_path)
    evaluator.train_baseline(prepared_data['training_df'])
    evaluator.train_gnn(
        prepared_data['data']['sensors'],
        prepared_data['time_windows'],
        prepared_data['labels'],
        epochs=30
    )
    evaluator.compare_models()
    evaluator.visualize_results()
    
    # Step 5: Interpretability analysis
    print("\n" + "=" * 80)
    print("STEP 5: Interpretability Analysis")
    print("=" * 80)
    interpreter = ModelInterpretability(output_dir=output_dir)
    
    config = load_dataset_config()
    physics = PhysicsModel(config)
    em = dataset['modulus_result']['elastic_modulus']
    lc_result = physics.compute_loading_cycles(em, np.arange(len(em)))
    loading_cycles = lc_result['loading_cycles']
    
    interpreter.generate_summary_report(
        baseline,
        dataset['modulus_result']['elastic_modulus'],
        loading_cycles
    )
    
    # Analyze spatial patterns
    cycle_indices = dataset['modulus_result'].get('cycle_indices', [])
    if len(cycle_indices) > 0:
        interpreter.analyze_spatial_damage_pattern(
            prepared_data['data']['sensors'],
            cycle_indices,
            0
        )
        if len(cycle_indices) > 1:
            interpreter.analyze_spatial_damage_pattern(
                prepared_data['data']['sensors'],
                cycle_indices,
                len(cycle_indices) - 1
            )
    
    print("\n" + "=" * 80)
    print("Full Pipeline Complete!")
    print("=" * 80)
    print(f"\nAll results saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - training_dataset.csv")
    print("  - physics_results.pkl")
    print("  - models/baseline_lightgbm.pkl")
    print("  - models/spatiotemporal_gnn.pth")
    print("  - model_comparison.png")
    print("  - baseline_feature_importance.png")
    print("  - modulus_degradation.png")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run full pipeline')
    parser.add_argument('--data_path', type=str, default='Input/raw/data.mat',
                       help='Path to raw .mat data file')
    parser.add_argument('--output_dir', type=str, default='Output',
                       help='Directory to save all outputs')
    
    args = parser.parse_args()
    
    run_full_pipeline(args.data_path, args.output_dir)

