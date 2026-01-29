import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional
import os

from Libs.models.networks.baselines.baseline import BaselineModel
from Libs.models.networks.spatiotemporal_gnn import SpatioTemporalGNN


class ModelInterpretability:
    """
    Provides interpretability analysis for fatigue life prediction models.
    """
    
    def __init__(self, output_dir='Output'):
        """
        Initialize the interpretability analyzer.
        
        Args:
            output_dir: Directory to save analysis results.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def analyze_baseline_features(self, baseline_model: BaselineModel):
        """
        Analyzes feature importance for the baseline model.
        
        Args:
            baseline_model: Trained baseline model.
        
        Returns:
            DataFrame with feature importance.
        """
        importance_df = baseline_model.get_feature_importance()
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.title('Baseline Model Feature Importance')
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'baseline_feature_importance.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {output_path}")
        plt.close()
        
        return importance_df
    
    def analyze_elastic_modulus_trend(self, elastic_modulus: np.ndarray,
                                      loading_cycles: Optional[np.ndarray] = None):
        """
        Analyzes the trend of elastic modulus over loading cycles.
        
        Args:
            elastic_modulus: Array of elastic modulus values.
            loading_cycles: Optional array of loading cycle counts.
        
        Returns:
            Dictionary with trend analysis results.
        """
        valid_mask = ~np.isnan(elastic_modulus)
        valid_modulus = elastic_modulus[valid_mask]
        
        if loading_cycles is not None:
            valid_cycles = loading_cycles[valid_mask]
        else:
            valid_cycles = np.arange(len(valid_modulus))
        
        # Compute degradation rate
        if len(valid_modulus) > 1:
            degradation_rate = np.diff(valid_modulus) / valid_modulus[:-1]
            avg_degradation_rate = np.mean(degradation_rate)
        else:
            avg_degradation_rate = 0
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        plt.plot(valid_cycles, valid_modulus, 'b-o', markersize=4, linewidth=2, label='Elastic Modulus')
        
        # Add initial modulus line
        initial_modulus = np.nanmean(valid_modulus[:min(10, len(valid_modulus))])
        plt.axhline(y=initial_modulus, color='g', linestyle='--', label=f'Initial Modulus ({initial_modulus:.2f})')
        
        # Add failure threshold line (50% of initial)
        failure_threshold = initial_modulus * 0.5
        plt.axhline(y=failure_threshold, color='r', linestyle='--', label=f'Failure Threshold ({failure_threshold:.2f})')
        
        plt.xlabel('Loading Cycles')
        plt.ylabel('Elastic Modulus')
        plt.title('Elastic Modulus Degradation Over Loading Cycles')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'modulus_degradation.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Modulus degradation plot saved to {output_path}")
        plt.close()
        
        return {
            'initial_modulus': initial_modulus,
            'final_modulus': valid_modulus[-1] if len(valid_modulus) > 0 else np.nan,
            'degradation_rate': avg_degradation_rate,
            'failure_threshold': failure_threshold
        }
    
    def analyze_spatial_damage_pattern(self, sensor_data_dict: Dict[str, pd.DataFrame],
                                      cycle_indices: list, cycle_idx: int):
        """
        Analyzes spatial damage pattern across sensors for a specific cycle.
        
        Args:
            sensor_data_dict: Dictionary of sensor data.
            cycle_indices: List of cycle index tuples.
            cycle_idx: Index of the cycle to analyze.
        
        Returns:
            Dictionary with spatial analysis results.
        """
        if cycle_idx >= len(cycle_indices):
            return None
        
        start_idx, end_idx = cycle_indices[cycle_idx]
        sensor_ids = sorted([int(k.split('_')[1]) for k in sensor_data_dict.keys()])
        
        # Extract force and displacement for each sensor
        from Libs.config import load_dataset_config
        
        config = load_dataset_config()
        sensors_config = config.get('sensors', {})
        sensor_positions_dict = sensors_config.get('sensor_positions', {})
        
        sensor_forces = []
        sensor_displacements = []
        sensor_positions = []
        
        for sid in sensor_ids:
            key = f'sensor_{sid}'
            sensor_df = sensor_data_dict[key]
            window_data = sensor_df.iloc[start_idx:end_idx]
            
            if len(window_data) > 0:
                pos = sensor_positions_dict.get(sid, sensor_positions_dict.get(str(sid), 0))
                pos = float(pos)
                if pos > 0:
                    sensor_forces.append(window_data['force'].mean())
                    sensor_displacements.append(window_data['displacement'].mean())
                    sensor_positions.append(pos)
        
        # Create visualization
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Force distribution
        axes[0].plot(sensor_positions, sensor_forces, 'b-o', linewidth=2, markersize=8)
        axes[0].set_xlabel('Sensor Position (mm)')
        axes[0].set_ylabel('Average Force (N)')
        axes[0].set_title(f'Spatial Force Distribution - Cycle {cycle_idx}')
        axes[0].grid(True, alpha=0.3)
        
        # Displacement distribution
        axes[1].plot(sensor_positions, sensor_displacements, 'r-o', linewidth=2, markersize=8)
        axes[1].set_xlabel('Sensor Position (mm)')
        axes[1].set_ylabel('Average Displacement (mm)')
        axes[1].set_title(f'Spatial Displacement Distribution - Cycle {cycle_idx}')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f'spatial_pattern_cycle_{cycle_idx}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Spatial pattern plot saved to {output_path}")
        plt.close()
        
        return {
            'sensor_positions': sensor_positions,
            'forces': sensor_forces,
            'displacements': sensor_displacements
        }
    
    def generate_summary_report(self, baseline_model: Optional[BaselineModel],
                               elastic_modulus: np.ndarray,
                               loading_cycles: Optional[np.ndarray] = None):
        """
        Generates a comprehensive interpretability summary report.
        
        Args:
            baseline_model: Trained baseline model (optional).
            elastic_modulus: Array of elastic modulus values.
            loading_cycles: Optional array of loading cycle counts.
        
        Returns:
            Dictionary with summary statistics.
        """
        report = {}
        
        # Baseline model analysis
        if baseline_model is not None:
            importance_df = self.analyze_baseline_features(baseline_model)
            report['feature_importance'] = importance_df.to_dict('records')
            print("\nTop 5 Most Important Features:")
            print(importance_df.head(5))
        
        # Modulus trend analysis
        trend_analysis = self.analyze_elastic_modulus_trend(elastic_modulus, loading_cycles)
        report['modulus_trend'] = trend_analysis
        
        print("\nModulus Degradation Analysis:")
        print(f"Initial Modulus: {trend_analysis['initial_modulus']:.2f}")
        print(f"Final Modulus: {trend_analysis['final_modulus']:.2f}")
        print(f"Average Degradation Rate: {trend_analysis['degradation_rate']:.4f}")
        print(f"Failure Threshold: {trend_analysis['failure_threshold']:.2f}")
        
        return report


if __name__ == '__main__':
    from Libs.data.dataloader import load_arrows_data
    from Libs.data.physics_processor import PhysicsModel
    from Libs.data.label_generator import FatigueLabelGenerator
    from Libs.models.networks.baselines.baseline import BaselineModel
    
    print("=" * 60)
    print("Model Interpretability Analysis")
    print("=" * 60)
    
    # Load and prepare data
    data = load_arrows_data('Input/raw/data.mat')
    physics = PhysicsModel()
    modulus_result = physics.compute_elastic_modulus(data['sensors'], data['time'])
    
    label_gen = FatigueLabelGenerator()
    training_df = label_gen.prepare_training_data(
        modulus_result, 
        data['sensors'], 
        data['time']
    )
    
    # Train baseline model
    baseline = BaselineModel(model_type='lightgbm')
    baseline.train(training_df, test_size=0.3)
    
    # Run interpretability analysis
    interpreter = ModelInterpretability()
    
    # Compute loading cycles from config (via PhysicsModel)
    lc_result = physics.compute_loading_cycles(modulus_result['elastic_modulus'], data['time'])
    loading_cycles = lc_result['loading_cycles']
    
    # Generate summary report
    report = interpreter.generate_summary_report(
        baseline,
        modulus_result['elastic_modulus'],
        loading_cycles
    )
    
    # Analyze spatial patterns for first and last cycles
    cycle_indices = modulus_result.get('cycle_indices', [])
    if len(cycle_indices) > 0:
        print("\nAnalyzing spatial damage patterns...")
        interpreter.analyze_spatial_damage_pattern(
            data['sensors'],
            cycle_indices,
            0  # First cycle
        )
        
        if len(cycle_indices) > 1:
            interpreter.analyze_spatial_damage_pattern(
                data['sensors'],
                cycle_indices,
                len(cycle_indices) - 1  # Last cycle
            )
    
    print("\n" + "=" * 60)
    print("Interpretability Analysis Complete!")
    print("=" * 60)

