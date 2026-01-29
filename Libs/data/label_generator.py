import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import uniform_filter1d
from typing import Dict, Any, Tuple, Optional


class FatigueLabelGenerator:
    """
    Generates fatigue life labels from elastic modulus time series.
    
    Fatigue failure is defined as when stiffness drops to a configured fraction
    of the initial stiffness estimated by a robust statistic.
    """
    
    def __init__(self, dataset_config=None, failure_threshold=0.5, initial_periods=10):
        """
        Initialize the label generator using YAML-managed parameters only.
        
        Args:
            dataset_config: Optional dataset configuration dictionary.
            failure_threshold: Fraction of initial modulus that defines failure (default 0.5).
            initial_periods: Number of initial periods to use for computing E_initial.
        """
        fatigue_config = {}
        if dataset_config is not None:
            fatigue_config = dataset_config.get('preprocessing', {}).get('fatigue_life', {})
            failure_threshold = float(fatigue_config.get('failure_threshold', failure_threshold))
            initial_periods = int(fatigue_config.get('initial_periods', initial_periods))
            self.min_failure_cycle_ratio = float(fatigue_config.get('min_failure_cycle_ratio', 0.2))
            self.enable_adaptive_threshold = bool(fatigue_config.get('enable_adaptive_threshold', False))
            self.initial_statistic = fatigue_config.get('initial_statistic', 'trimmed_mean')
            self.trim_fraction = float(fatigue_config.get('trim_fraction', 0.1))
            self.min_valid_initial_points = int(fatigue_config.get('min_valid_initial_points', max(3, initial_periods // 2)))
            self.smoothing_window = int(fatigue_config.get('smoothing_window', 5))
            self.consecutive_below_threshold = int(fatigue_config.get('consecutive_below_threshold', 3))
            self.search_start_ratio = float(fatigue_config.get('search_start_ratio', self.min_failure_cycle_ratio))
            self.loading_cycles_offset = int(fatigue_config.get('loading_cycles_offset', 200))
            self.loading_cycles_per_step = int(fatigue_config.get('loading_cycles_per_step', 3000))
            self.rul_transform = fatigue_config.get('rul_transform')
        else:
            self.min_failure_cycle_ratio = 0.2
            self.enable_adaptive_threshold = False
            self.initial_statistic = 'trimmed_mean'
            self.trim_fraction = 0.1
            self.min_valid_initial_points = max(3, initial_periods // 2)
            self.smoothing_window = 5
            self.consecutive_below_threshold = 3
            self.search_start_ratio = self.min_failure_cycle_ratio
            self.loading_cycles_offset = 200
            self.loading_cycles_per_step = 3000
            self.rul_transform = None
        
        self.failure_threshold = float(failure_threshold)
        self.initial_periods = int(initial_periods)
        self._fatigue_config = fatigue_config
    
    def _smooth_modulus(self, elastic_modulus: np.ndarray) -> np.ndarray:
        """
        Smooths elastic modulus to suppress high-frequency noise before failure search.
        
        Uses a uniform filter with configurable window size; falls back to raw values when
        smoothing is effectively disabled.
        """
        if self.smoothing_window <= 1:
            return elastic_modulus
        return uniform_filter1d(elastic_modulus, size=self.smoothing_window, mode='nearest')
    
    def compute_initial_modulus(self, elastic_modulus: np.ndarray) -> float:
        """
        Computes a robust initial elastic modulus from the first periods.
        
        Uses trimmed mean or median based on configuration to reduce sensitivity
        to early noise/outliers.
        """
        valid_modulus = elastic_modulus[~np.isnan(elastic_modulus)]
        if len(valid_modulus) == 0:
            return np.nan
        
        window = valid_modulus[: min(len(valid_modulus), self.initial_periods)]
        if len(window) < self.min_valid_initial_points:
            return float(np.nanmedian(window))
        
        if self.initial_statistic == 'median':
            return float(np.nanmedian(window))
        
        trimmed = stats.trim_mean(window, self.trim_fraction)
        return float(trimmed)
    
    def find_fatigue_failure(self, elastic_modulus: np.ndarray, 
                            loading_cycles: Optional[np.ndarray] = None) -> Tuple[int, float]:
        """
        Finds the fatigue failure point where modulus drops below threshold.
        
        Includes diagnostic information and adaptive threshold adjustment if failure occurs too early.
        
        Args:
            elastic_modulus: Array of elastic modulus values E(t).
            loading_cycles: Optional array of loading cycle counts corresponding to each modulus value.
        
        Returns:
            Tuple of (failure_index, failure_cycle) where:
            - failure_index: Index where failure occurs (or -1 if not found).
            - failure_cycle: Loading cycle at failure (or NaN if not found).
        """
        smoothed_modulus = self._smooth_modulus(elastic_modulus)
        E_initial = self.compute_initial_modulus(smoothed_modulus)
        
        if np.isnan(E_initial) or E_initial <= 0:
            print(f"[Diagnostic] find_fatigue_failure: Invalid E_initial={E_initial}")
            return -1, np.nan
        
        valid_mask = ~np.isnan(smoothed_modulus)
        valid_modulus = smoothed_modulus[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        n_valid = len(valid_modulus)
        
        failure_threshold_value = E_initial * self.failure_threshold
        
        print(f"[Diagnostic] find_fatigue_failure: E_initial={E_initial:.4e}, failure_threshold_value={failure_threshold_value:.4e} ({self.failure_threshold*100:.1f}% of initial)")
        
        below_threshold = valid_modulus < failure_threshold_value
        if not np.any(below_threshold):
            print(f"[Diagnostic] find_fatigue_failure: No failure point found (all modulus values above threshold)")
            return -1, np.nan
        
        start_idx = int(np.floor(n_valid * self.search_start_ratio))
        kernel = np.ones(max(1, self.consecutive_below_threshold), dtype=int)
        run_lengths = np.convolve(below_threshold.astype(int), kernel, mode='valid')
        candidate_indices = np.where(run_lengths == kernel.size)[0]
        candidate_indices = candidate_indices[candidate_indices >= start_idx] if len(candidate_indices) > 0 else candidate_indices
        
        if len(candidate_indices) == 0:
            print(f"[Diagnostic] find_fatigue_failure: No sustained failure detected after {self.search_start_ratio*100:.1f}% of cycles")
            return -1, np.nan
        
        failure_idx_in_valid = int(candidate_indices[0])
        failure_idx = valid_indices[failure_idx_in_valid]
        failure_ratio = failure_idx / n_valid if n_valid > 0 else 0
        
        failure_cycle = loading_cycles[failure_idx] if loading_cycles is not None and len(loading_cycles) > failure_idx else np.nan
        failure_modulus = valid_modulus[failure_idx_in_valid]
        
        print(f"[Diagnostic] find_fatigue_failure: failure_idx={failure_idx}, failure_ratio={failure_ratio:.3f}, failure_modulus={failure_modulus:.4e}")
        return failure_idx, failure_cycle
    
    def generate_labels(self, elastic_modulus: np.ndarray,
                       loading_cycles: Optional[np.ndarray] = None,
                       cycle_indices: Optional[list] = None) -> Dict[str, Any]:
        """
        Generates fatigue life labels for the dataset.
        
        Args:
            elastic_modulus: Array of elastic modulus values E(t) for each loading cycle.
            loading_cycles: Optional array of loading cycle counts.
            cycle_indices: Optional list of (start_idx, end_idx) tuples for each cycle.
        
        Returns:
            Dictionary containing:
            - 'fatigue_life': Total fatigue life (N_f) in cycles.
            - 'failure_index': Index where failure occurs.
            - 'initial_modulus': Initial elastic modulus E_initial.
            - 'failure_modulus': Modulus at failure point.
            - 'rul': Array of remaining useful life (RUL) for each time point.
            - 'labels': Array of labels (RUL values) for each cycle.
        """
        E_initial = self.compute_initial_modulus(elastic_modulus)
        failure_idx, failure_cycle = self.find_fatigue_failure(elastic_modulus, loading_cycles)
        
        n_cycles = len(elastic_modulus)
        rul = np.full(n_cycles, np.nan)
        if failure_idx >= 0:
            if loading_cycles is not None and len(loading_cycles) >= n_cycles:
                remaining = np.maximum(0, failure_cycle - loading_cycles[:n_cycles])
                remaining = np.clip(remaining, a_min=0, a_max=None)
                rul[:n_cycles] = remaining
                rul[failure_idx + 1 :] = 0
            else:
                remaining = np.maximum(0, failure_idx - np.arange(n_cycles))
                rul[:n_cycles] = remaining
                rul[failure_idx + 1 :] = 0
        
        result = {
            'fatigue_life': failure_cycle if not np.isnan(failure_cycle) else failure_idx,
            'failure_index': failure_idx,
            'initial_modulus': E_initial,
            'failure_modulus': elastic_modulus[failure_idx] if failure_idx >= 0 else np.nan,
            'rul': rul,
            'labels': rul,
            'elastic_modulus': elastic_modulus,
            'loading_cycles': loading_cycles
        }
        
        # Diagnostic information
        rul_nan_count = np.sum(np.isnan(rul))
        rul_valid_count = n_cycles - rul_nan_count
        failure_found = failure_idx >= 0
        
        if failure_found:
            pre_failure_count = np.sum((rul > 0) & (~np.isnan(rul)))
            post_failure_count = np.sum((rul == 0) & (~np.isnan(rul)))
            rul_values = rul[~np.isnan(rul)]
            
            print(f"[Diagnostic] generate_labels: failure_idx={failure_idx}, failure_found={failure_found}")
            print(f"[Diagnostic] generate_labels: E_initial={E_initial:.4e}, failure_modulus={result['failure_modulus']:.4e}")
            print(f"[Diagnostic] generate_labels: RUL NaN count={rul_nan_count}, RUL valid count={rul_valid_count}")
            print(f"[Diagnostic] generate_labels: Pre-failure samples={pre_failure_count}, Post-failure samples={post_failure_count}")
            
            if len(rul_values) > 0:
                print(f"[Diagnostic] generate_labels: RUL statistics - min={np.min(rul_values):.2f}, max={np.max(rul_values):.2f}, mean={np.mean(rul_values):.2f}, median={np.median(rul_values):.2f}")
            
            if pre_failure_count < n_cycles * 0.2:
                print(f"[Warning] generate_labels: Very few pre-failure samples ({pre_failure_count}/{n_cycles}), model may struggle to learn degradation pattern")
            elif post_failure_count > pre_failure_count * 2:
                print(f"[Warning] generate_labels: Imbalanced dataset - post-failure samples ({post_failure_count}) >> pre-failure samples ({pre_failure_count})")
        else:
            print(f"[Diagnostic] generate_labels: No failure point found, all RUL values are NaN")
        
        return result
    
    def prepare_training_data(self, physics_result: Dict[str, Any],
                             sensor_data_dict: Dict[str, pd.DataFrame],
                             time_data: np.ndarray) -> pd.DataFrame:
        """
        Prepares training dataset with features and labels.
        
        Args:
            physics_result: Result dictionary from PhysicsModel.compute_elastic_modulus().
            sensor_data_dict: Dictionary of sensor data.
            time_data: Array of time values.
        
        Returns:
            DataFrame with features and labels for training.
        """
        elastic_modulus = physics_result['elastic_modulus']
        cycle_indices = physics_result.get('cycle_indices', [])
        
        # Compute loading cycles from YAML-configured offset and per-step
        n_cycles = len(elastic_modulus)
        loading_cycles = self.loading_cycles_offset + np.arange(n_cycles) * self.loading_cycles_per_step
        
        # Generate labels
        labels = self.generate_labels(elastic_modulus, loading_cycles, cycle_indices)
        
        # Prepare features
        features_list = []
        
        for i in range(n_cycles):
            if np.isnan(elastic_modulus[i]):
                continue
            
            rul_val = float(labels['rul'][i])
            if self.rul_transform == 'log1p' and np.isfinite(rul_val):
                rul_val = np.log1p(rul_val)
            feature_dict = {
                'cycle_index': i,
                'loading_cycle': loading_cycles[i],
                'elastic_modulus': elastic_modulus[i],
                'stiffness': physics_result['stiffness'][i],
                'rul': rul_val,
                'fatigue_life': labels['fatigue_life']
            }
            
            # Add Gaussian parameters if available
            if i < len(physics_result['gaussian_params']):
                gauss_params = physics_result['gaussian_params'][i]
                if gauss_params.get('disp') is not None:
                    feature_dict['disp_a'] = gauss_params['disp'][0]
                    feature_dict['disp_b'] = gauss_params['disp'][1]
                    feature_dict['disp_c'] = gauss_params['disp'][2]
                if gauss_params.get('force') is not None:
                    feature_dict['force_a'] = gauss_params['force'][0]
                    feature_dict['force_b'] = gauss_params['force'][1]
                    feature_dict['force_c'] = gauss_params['force'][2]
            
            # Compute modulus change rate (if previous cycle exists)
            if i > 0 and not np.isnan(elastic_modulus[i-1]):
                feature_dict['modulus_change_rate'] = (
                    elastic_modulus[i] - elastic_modulus[i-1]
                ) / elastic_modulus[i-1]
            else:
                feature_dict['modulus_change_rate'] = 0
            
            features_list.append(feature_dict)
        
        df = pd.DataFrame(features_list)
        
        # Diagnostic information
        skipped_count = n_cycles - len(features_list)
        print(f"[Diagnostic] prepare_training_data: {n_cycles} cycles total, {skipped_count} skipped (NaN elastic_modulus), {len(df)} rows in DataFrame")
        
        return df


if __name__ == '__main__':
    from Libs.dataloader import load_arrows_data
    from Libs.physics_model import PhysicsModel
    
    print("Loading ARROWS data...")
    data = load_arrows_data('Input/raw/data.mat')
    
    if data:
        print("Computing elastic modulus...")
        physics = PhysicsModel()
        modulus_result = physics.compute_elastic_modulus(data['sensors'], data['time'])
        
        print("Generating fatigue life labels...")
        label_gen = FatigueLabelGenerator()
        training_df = label_gen.prepare_training_data(
            modulus_result, 
            data['sensors'], 
            data['time']
        )
        
        print(f"\nTraining dataset shape: {training_df.shape}")
        print(f"Features: {list(training_df.columns)}")
        print(f"\nFatigue life: {training_df['fatigue_life'].iloc[0]:.0f} cycles")
        print(f"Initial modulus: {label_gen.compute_initial_modulus(modulus_result['elastic_modulus']):.2f}")
        print(f"Failure index: {label_gen.find_fatigue_failure(modulus_result['elastic_modulus'])[0]}")
        print("\nSample data:")
        print(training_df.head())

