import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any, Tuple
import os

from Libs.data.dataloader import load_arrows_data
from Libs.data.physics_processor import PhysicsModel
from Libs.data.label_generator import FatigueLabelGenerator
from Libs.models.networks.baselines.baseline import BaselineModel
from Libs.models.networks.spatiotemporal_gnn import SpatioTemporalGNN


class ModelEvaluator:
    """
    Evaluates and compares baseline and GNN models for fatigue life prediction.
    """
    
    def __init__(self, output_dir='Output'):
        """
        Initialize the evaluator.
        
        Args:
            output_dir: Directory to save evaluation results.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.baseline_model = None
        self.gnn_model = None
        self.results = {}
    
    def prepare_data(self, data_path='Input/raw/data.mat'):
        """
        Prepares all data for training and evaluation.
        
        Args:
            data_path: Path to the data file.
        
        Returns:
            Dictionary containing prepared data.
        """
        print("=" * 60)
        print("Step 1: Loading and preprocessing data...")
        print("=" * 60)
        
        # Load data
        data = load_arrows_data(data_path)
        
        # Compute elastic modulus
        print("\nComputing elastic modulus...")
        physics = PhysicsModel()
        modulus_result = physics.compute_elastic_modulus(data['sensors'], data['time'])
        
        # Generate labels
        print("\nGenerating fatigue life labels...")
        label_gen = FatigueLabelGenerator()
        training_df = label_gen.prepare_training_data(
            modulus_result, 
            data['sensors'], 
            data['time']
        )
        
        # Prepare graph data for GNN
        print("\nPreparing graph data for GNN...")
        cycle_indices = modulus_result.get('cycle_indices', [])
        labels = training_df['rul'].values
        
        valid_mask = ~np.isnan(labels)
        valid_indices = np.where(valid_mask)[0]
        
        time_windows = []
        valid_labels = []
        
        for idx in valid_indices:
            if idx < len(cycle_indices):
                time_windows.append(cycle_indices[idx])
                valid_labels.append(labels[idx])
        
        return {
            'data': data,
            'modulus_result': modulus_result,
            'training_df': training_df,
            'time_windows': time_windows,
            'labels': np.array(valid_labels),
            'valid_indices': valid_indices
        }
    
    def train_baseline(self, training_df: pd.DataFrame, test_size=0.3):
        """
        Trains the baseline model.
        
        Args:
            training_df: Training dataframe.
            test_size: Fraction of data for testing.
        
        Returns:
            Evaluation results.
        """
        print("\n" + "=" * 60)
        print("Step 2: Training Baseline Model (LightGBM)...")
        print("=" * 60)
        
        self.baseline_model = BaselineModel(model_type='lightgbm')
        results = self.baseline_model.train(training_df, test_size=test_size)
        
        self.results['baseline'] = results
        return results
    
    def train_gnn(self, sensor_data_dict: Dict, time_windows: list, 
                  labels: np.ndarray, test_size=0.3, epochs=50):
        """
        Trains the GNN model.
        
        Args:
            sensor_data_dict: Dictionary of sensor data.
            time_windows: List of time window tuples.
            labels: Array of labels.
            test_size: Fraction of data for testing.
            epochs: Number of training epochs.
        
        Returns:
            Evaluation results.
        """
        print("\n" + "=" * 60)
        print("Step 3: Training GNN Model...")
        print("=" * 60)
        
        # Split data
        n_samples = len(time_windows)
        indices = np.arange(n_samples)
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=42
        )
        
        train_windows = [time_windows[i] for i in train_idx]
        test_windows = [time_windows[i] for i in test_idx]
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]
        
        # Prepare graph data
        self.gnn_model = SpatioTemporalGNN(num_nodes=9, input_dim=2, hidden_dim=32)
        
        train_graph_data = self.gnn_model.prepare_graph_data(
            sensor_data_dict, train_windows, train_labels
        )
        test_graph_data = self.gnn_model.prepare_graph_data(
            sensor_data_dict, test_windows, test_labels
        )
        
        print(f"Training on {len(train_graph_data)} samples...")
        
        # Train model
        loss_history = self.gnn_model.train(
            train_graph_data, epochs=epochs, lr=0.001, batch_size=4, verbose=True
        )
        
        # Evaluate
        train_pred = self.gnn_model.predict(train_graph_data)
        test_pred = self.gnn_model.predict(test_graph_data)
        
        train_rmse = np.sqrt(mean_squared_error(train_labels, train_pred))
        test_rmse = np.sqrt(mean_squared_error(test_labels, test_pred))
        train_mae = mean_absolute_error(train_labels, train_pred)
        test_mae = mean_absolute_error(test_labels, test_pred)
        train_r2 = r2_score(train_labels, train_pred)
        test_r2 = r2_score(test_labels, test_pred)
        mask_pos = test_labels > 0
        n_pos = int(np.sum(mask_pos))
        if n_pos >= 2:
            ss_pos = np.sum((test_labels[mask_pos] - np.mean(test_labels[mask_pos])) ** 2)
            test_r2_rul_positive = np.nan if ss_pos == 0 else float(r2_score(test_labels[mask_pos], test_pred[mask_pos]))
        else:
            test_r2_rul_positive = np.nan
        
        print(f"\nGNN Model Performance:")
        print(f"Train RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}")
        print(f"Train MAE: {train_mae:.2f}, Test MAE: {test_mae:.2f}")
        print(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        print(f"Test R² (RUL>0): {test_r2_rul_positive:.4f}" if not np.isnan(test_r2_rul_positive) else "Test R² (RUL>0): NaN")
        
        results = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_r2_rul_positive': test_r2_rul_positive,
            'y_test': test_labels,
            'y_pred': test_pred,
            'loss_history': loss_history
        }
        
        self.results['gnn'] = results
        return results
    
    def compare_models(self):
        """
        Compares baseline and GNN model performance.
        
        Returns:
            Comparison results dictionary.
        """
        print("\n" + "=" * 60)
        print("Step 4: Model Comparison")
        print("=" * 60)
        
        if 'baseline' not in self.results or 'gnn' not in self.results:
            print("Error: Both models must be trained before comparison.")
            return None
        
        baseline = self.results['baseline']
        gnn = self.results['gnn']
        
        comparison = {
            'baseline': {
                'test_rmse': baseline['test_rmse'],
                'test_mae': baseline['test_mae'],
                'test_r2': baseline['test_r2'],
                'test_r2_rul_positive': baseline.get('test_r2_rul_positive'),
            },
            'gnn': {
                'test_rmse': gnn['test_rmse'],
                'test_mae': gnn['test_mae'],
                'test_r2': gnn['test_r2'],
                'test_r2_rul_positive': gnn.get('test_r2_rul_positive'),
            }
        }
        
        print("\nPerformance Comparison:")
        print("-" * 60)
        print(f"{'Metric':<20} {'Baseline':<20} {'GNN':<20}")
        print("-" * 60)
        print(f"{'Test RMSE':<20} {baseline['test_rmse']:<20.4f} {gnn['test_rmse']:<20.4f}")
        print(f"{'Test MAE':<20} {baseline['test_mae']:<20.4f} {gnn['test_mae']:<20.4f}")
        print(f"{'Test R²':<20} {baseline['test_r2']:<20.4f} {gnn['test_r2']:<20.4f}")
        b_r2p = comparison['baseline']['test_r2_rul_positive']
        g_r2p = comparison['gnn']['test_r2_rul_positive']
        b_r2p_s = f"{b_r2p:.4f}" if b_r2p is not None and not (isinstance(b_r2p, float) and np.isnan(b_r2p)) else "N/A"
        g_r2p_s = f"{g_r2p:.4f}" if g_r2p is not None and not (isinstance(g_r2p, float) and np.isnan(g_r2p)) else "N/A"
        print(f"{'Test R² (RUL>0)':<20} {b_r2p_s:<20} {g_r2p_s:<20}")
        print("-" * 60)
        
        # Determine winner
        if gnn['test_rmse'] < baseline['test_rmse']:
            print("\n✓ GNN model performs better (lower RMSE)")
        elif baseline['test_rmse'] < gnn['test_rmse']:
            print("\n✓ Baseline model performs better (lower RMSE)")
        else:
            print("\nModels perform similarly")
        
        self.results['comparison'] = comparison
        return comparison
    
    def visualize_results(self):
        """
        Creates visualization plots comparing model predictions.
        """
        if 'baseline' not in self.results or 'gnn' not in self.results:
            print("Error: Both models must be trained before visualization.")
            return
        
        baseline = self.results['baseline']
        gnn = self.results['gnn']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Baseline predictions
        axes[0].scatter(baseline['y_test'], baseline['y_pred'], alpha=0.6)
        axes[0].plot([baseline['y_test'].min(), baseline['y_test'].max()],
                     [baseline['y_test'].min(), baseline['y_test'].max()], 'r--', lw=2)
        axes[0].set_xlabel('True RUL')
        axes[0].set_ylabel('Predicted RUL')
        axes[0].set_title(f'Baseline Model (R² = {baseline["test_r2"]:.4f})')
        axes[0].grid(True, alpha=0.3)
        
        # GNN predictions
        axes[1].scatter(gnn['y_test'], gnn['y_pred'], alpha=0.6, color='green')
        axes[1].plot([gnn['y_test'].min(), gnn['y_test'].max()],
                     [gnn['y_test'].min(), gnn['y_test'].max()], 'r--', lw=2)
        axes[1].set_xlabel('True RUL')
        axes[1].set_ylabel('Predicted RUL')
        axes[1].set_title(f'GNN Model (R² = {gnn["test_r2"]:.4f})')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'model_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to {output_path}")
        plt.close()


if __name__ == '__main__':
    evaluator = ModelEvaluator()
    
    # Prepare data
    prepared_data = evaluator.prepare_data()
    
    # Train baseline model
    baseline_results = evaluator.train_baseline(prepared_data['training_df'])
    
    # Train GNN model
    gnn_results = evaluator.train_gnn(
        prepared_data['data']['sensors'],
        prepared_data['time_windows'],
        prepared_data['labels'],
        epochs=30
    )
    
    # Compare models
    comparison = evaluator.compare_models()
    
    # Visualize results
    evaluator.visualize_results()
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)

