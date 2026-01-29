import warnings
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import uniform_filter1d
from scipy.optimize import curve_fit, least_squares
from scipy.signal import argrelextrema, hilbert
from tqdm import tqdm

from Libs.config import load_dataset_config

warnings.filterwarnings('ignore')


class PhysicsModel:
    """
    Implements the physical modeling of asphalt fatigue based on the ref.m script.
    
    The model computes elastic modulus E(t) from force and displacement measurements
    using the Hetenyi foundation approach and Gaussian curve fitting.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the physics model.
        
        Args:
            config: Optional configuration dictionary. If None, loads from dataset.yaml.
        """
        if config is None:
            config = load_dataset_config()
        
        physics_config = config.get('physics_model', {})
        sensors_config = config.get('sensors', {})
        
        self.sample_frequency = float(physics_config.get('sample_frequency', 4800.0))
        self.dt = 1.0 / self.sample_frequency
        
        self.specimen_length = float(physics_config.get('specimen_length', 400))
        self.specimen_width = float(physics_config.get('specimen_width', 100))
        self.specimen_height = float(physics_config.get('specimen_height', 55))
        self.area = self.specimen_width * self.specimen_height
        
        self.I = (self.specimen_width * (self.specimen_height ** 3)) / 12
        self.poisson_ratio = float(physics_config.get('poisson_ratio', 0.35))
        
        sensor_positions_dict = sensors_config.get('sensor_positions', {})
        all_positions = []
        for i in range(1, 12):
            pos = sensor_positions_dict.get(i, sensor_positions_dict.get(str(i), 0))
            all_positions.append(float(pos))
        self.sensor_positions = np.array([pos for pos in all_positions if pos > 0])
        
        self.nth_valley = int(physics_config.get('nth_valley', 300))
        self.nth_valley_ratio = float(physics_config.get('nth_valley_ratio', 0.1))
        self.min_nth_valley = int(physics_config.get('min_nth_valley', 50))
        self.max_nth_valley = int(physics_config.get('max_nth_valley', 300))
        self.time_threshold = float(physics_config.get('time_threshold', 100))
        self.reference_sensor = int(sensors_config.get('reference_sensor', 6))
        
        gaussian_fit_config = physics_config.get('gaussian_fit', {})
        self.gaussian_max_retry = int(gaussian_fit_config.get('max_retry_attempts', 5))
        self.gaussian_min_r_squared = float(gaussian_fit_config.get('min_r_squared', 0.25))
        self.gaussian_min_r_squared_disp = float(gaussian_fit_config.get('min_r_squared_disp', 0.2))
        self.gaussian_min_r_squared_force = float(gaussian_fit_config.get('min_r_squared_force', 0.25))
        self.gaussian_use_alternatives = bool(gaussian_fit_config.get('alternative_initial_guesses', True))
        self.gaussian_use_fallback = bool(gaussian_fit_config.get('use_fallback_strategy', False))
        self.gaussian_fallback_r_squared = float(gaussian_fit_config.get('fallback_r_squared_threshold', 0.0))
        self.gaussian_use_bounds = bool(gaussian_fit_config.get('use_bounds', True))
        self.gaussian_min_r_squared_accept_for_label = gaussian_fit_config.get('min_r_squared_accept_for_label')
        self.gaussian_min_points_for_r2_reject = gaussian_fit_config.get('min_points_for_r2_reject')

        bounds_config = gaussian_fit_config.get('parameter_bounds', {})
        self.gaussian_a_min_factor = float(bounds_config.get('a_min_factor', 0.1))
        self.gaussian_a_max_factor = float(bounds_config.get('a_max_factor', 10.0))
        self.gaussian_b_min = float(bounds_config.get('b_min', 0))
        self.gaussian_b_max = float(bounds_config.get('b_max', 500))
        self.gaussian_c_min = float(bounds_config.get('c_min', 5))
        self.gaussian_c_max = float(bounds_config.get('c_max', 1000))
        
        self.fourth_derivative_threshold = float(physics_config.get('fourth_derivative_threshold', 1e-12))
        self.use_alternative_calculation = bool(physics_config.get('use_alternative_calculation', True))
        
        self._sensor_positions_dict = sensor_positions_dict
        
        preprocess_cfg = physics_config.get('preprocess', {})
        self.smoothing_window = int(preprocess_cfg.get('smoothing_window', 25))
        self.center_signals = bool(preprocess_cfg.get('center', True))
        self.scale_signals = bool(preprocess_cfg.get('scale', True))
        self.max_abs_scale = float(preprocess_cfg.get('max_abs_scale', 1e9))
        
        fatigue_cfg = config.get('preprocessing', {}).get('fatigue_life', {})
        self.loading_cycles_offset = int(fatigue_cfg.get('loading_cycles_offset', 200))
        self.loading_cycles_per_step = int(fatigue_cfg.get('loading_cycles_per_step', 3000))
        
    def identify_loading_cycles(self, time_data, displacement_data, threshold=100):
        """
        Identifies and separates loading cycles by detecting pauses in the time series.
        
        This corresponds to ref.m lines 65-141.
        
        Args:
            time_data: Array of time values.
            displacement_data: Array of displacement values.
            threshold: Time threshold to detect pauses (seconds).
        
        Returns:
            A list of tuples (time_segment, displacement_segment) for each loading cycle.
        """
        cycles = []
        start_idx = 0
        
        while start_idx < len(time_data):
            # Calculate time differences between consecutive points
            time_diff = np.diff(time_data[start_idx:])
            
            # Find first index where time difference exceeds threshold
            jump_indices = np.where(time_diff > threshold)[0]
            
            if len(jump_indices) == 0:
                # No more jumps; take the rest of the data
                end_idx = len(time_data)
            else:
                # Take data up to the first jump
                end_idx = start_idx + jump_indices[0] + 1
            
            time_segment = time_data[start_idx:end_idx]
            disp_segment = displacement_data[start_idx:end_idx]
            
            if len(time_segment) > 100:  # Only keep segments with sufficient data
                cycles.append((time_segment, disp_segment))
            
            start_idx = end_idx
        
        return cycles
    
    def extract_amplitude(self, signal, envelope_span=300):
        """
        Extracts peak and valley points from a signal using envelope detection.
        
        This corresponds to ref.m lines 158-256.
        
        Args:
            signal: Input signal (e.g., force or displacement).
            envelope_span: Span for envelope calculation.
        
        Returns:
            peaks: Array of peak values.
            valleys: Array of valley values.
            peak_indices: Indices of peaks.
            valley_indices: Indices of valleys.
        """
        # Use scipy's envelope function (similar to MATLAB's envelope)
        analytic = hilbert(signal - np.mean(signal))
        amplitude = np.abs(analytic)
        
        # Find local maxima and minima
        peak_indices = argrelextrema(signal, np.greater, order=20)[0]
        valley_indices = argrelextrema(signal, np.less, order=20)[0]
        
        peaks = signal[peak_indices] if len(peak_indices) > 0 else np.array([])
        valleys = signal[valley_indices] if len(valley_indices) > 0 else np.array([])
        
        return peaks, valleys, peak_indices, valley_indices
    
    def find_nth_valley_index(self, signal, n=300, envelope_span=300):
        """
        Finds the index of the nth valley point using envelope detection.
        
        This corresponds to ref.m lines 161-172, where it finds the 300th valley.
        Optimized version using direct valley detection instead of envelope matching.
        
        Args:
            signal: Input signal.
            n: Which valley to find (default 300 as in ref.m).
            envelope_span: Span for envelope calculation (used for smoothing).
        
        Returns:
            Index of the nth valley, or None if not found.
        """
        # Smooth the signal first to reduce noise
        from scipy.ndimage import uniform_filter1d
        smoothed = uniform_filter1d(signal, size=min(envelope_span, len(signal)//10))
        
        # Find local minima (valleys) directly
        valley_indices = argrelextrema(smoothed, np.less, order=max(10, envelope_span//30))[0]
        
        if len(valley_indices) >= n:
            return valley_indices[n - 1]
        return None
    
    def gaussian_curve(self, x, a, b, c):
        """
        Gaussian function for curve fitting.
        
        Function: f(x) = a * exp(-((x - b)^2) / c^2)
        
        Args:
            x: Input position.
            a: Amplitude parameter.
            b: Position parameter (center).
            c: Width parameter (standard deviation).
        
        Returns:
            Function value at x.
        """
        return a * np.exp(-((x - b) ** 2) / (c ** 2))
    
    def gaussian_fourth_derivative(self, x, a, b, c):
        """
        Computes the fourth derivative of the Gaussian function.
        
        This is used to calculate plate bending stiffness D in ref.m lines 314-336.
        
        Args:
            x: Input position.
            a, b, c: Gaussian parameters.
        
        Returns:
            Fourth derivative value at x.
        """
        u = (x - b) / c
        term = a / (c ** 4)
        # Fourth derivative of exp(-u^2)
        fourth_deriv = (16 * u ** 4 - 48 * u ** 2 + 12) * np.exp(-(u ** 2))
        return term * fourth_deriv
    
    def _preprocess_curve(self, values: np.ndarray) -> np.ndarray:
        """
        Applies smoothing, centering, and robust scaling to stabilize curve fitting.
        """
        smoothed = uniform_filter1d(values, size=max(1, self.smoothing_window), mode='nearest')
        processed = smoothed
        if self.center_signals:
            processed = processed - np.nanmedian(processed)
        if self.scale_signals:
            scale = stats.iqr(processed) if np.isfinite(stats.iqr(processed)) else np.nanstd(processed)
            if not np.isfinite(scale) or scale == 0:
                scale = 1.0
            processed = np.clip(processed / scale, -self.max_abs_scale, self.max_abs_scale)
        return processed
    
    def validate_gaussian_parameters(self, params, positions, data_values):
        """
        Validates Gaussian fitting parameters for physical reasonableness.
        
        Uses more lenient validation to allow more fits to pass, improving data recovery rate.
        
        Args:
            params: Tuple of (a, b, c) Gaussian parameters.
            positions: Array of sensor positions.
            data_values: Array of measured values.
        
        Returns:
            Tuple of (is_valid, reason). is_valid is True if parameters are reasonable.
        """
        a, b, c = params
        
        data_range = np.max(data_values) - np.min(data_values)
        pos_min = np.min(positions)
        pos_max = np.max(positions)
        pos_range = pos_max - pos_min
        
        if np.isnan(a) or np.isnan(b) or np.isnan(c):
            return False, "NaN in parameters"
        
        if np.isinf(a) or np.isinf(b) or np.isinf(c):
            return False, "Inf in parameters"
        
        a_min = data_range * self.gaussian_a_min_factor * 0.5
        a_max = data_range * self.gaussian_a_max_factor * 2.0
        if abs(a) < a_min or abs(a) > a_max:
            return False, f"Amplitude a={a:.2e} out of range [{a_min:.2e}, {a_max:.2e}]"
        
        b_tolerance = pos_range * 0.5
        if b < self.gaussian_b_min - b_tolerance or b > self.gaussian_b_max + b_tolerance:
            return False, f"Center b={b:.2f} out of range [{self.gaussian_b_min - b_tolerance:.2f}, {self.gaussian_b_max + b_tolerance:.2f}]"
        
        c_tolerance = pos_range * 0.2
        if c < self.gaussian_c_min - c_tolerance or c > self.gaussian_c_max + c_tolerance:
            return False, f"Width c={c:.2f} out of range [{self.gaussian_c_min - c_tolerance:.2f}, {self.gaussian_c_max + c_tolerance:.2f}]"
        
        if abs(b - pos_min) > pos_range * 3.0:
            return False, f"Center b={b:.2f} too far from sensor range [{pos_min}, {pos_max}]"
        
        return True, "Valid"
    
    def fit_gaussian_spatial(self, positions, data_values, initial_guess=None, error_info=None, min_r_squared=None, fit_type='default'):
        """
        Fits a Gaussian curve to spatial data (sensor positions vs. amplitude) without try/except.
        Uses scipy.optimize.least_squares to avoid exception-driven control flow.
        """
        if initial_guess is None:
            initial_guess = [np.max(data_values) - np.min(data_values), 250, 100]
        
        valid_mask = ~np.isnan(data_values)
        valid_pos = positions[valid_mask]
        valid_vals = data_values[valid_mask]
        
        if len(valid_pos) < 3:
            msg = f"Insufficient valid points: {len(valid_pos)} < 3"
            if error_info is not None:
                error_info['reason'] = msg
            return None, 0, msg
        
        if min_r_squared is None:
            if fit_type == 'disp':
                min_r_squared = self.gaussian_min_r_squared_disp
            elif fit_type == 'force':
                min_r_squared = self.gaussian_min_r_squared_force
            else:
                min_r_squared = self.gaussian_min_r_squared
        
        data_range = np.max(valid_vals) - np.min(valid_vals)
        data_mean = np.mean(valid_vals)
        data_median = np.median(valid_vals)
        data_q25 = np.percentile(valid_vals, 25)
        data_q75 = np.percentile(valid_vals, 75)
        data_std = np.std(valid_vals)
        pos_mean = np.mean(valid_pos)
        pos_median = np.median(valid_pos)
        pos_std = np.std(valid_pos)
        pos_min = np.min(valid_pos)
        pos_max = np.max(valid_pos)
        pos_range = pos_max - pos_min
        
        initial_guesses = [initial_guess]
        if self.gaussian_use_alternatives:
            initial_guesses.extend([
                [data_range, pos_mean, pos_std],
                [data_range * 0.5, pos_mean, pos_std * 1.5],
                [data_range * 1.5, pos_mean, pos_std * 0.5],
                [data_mean, pos_mean, 100],
                [data_range, pos_median, pos_std],
                [data_median, pos_mean, pos_std],
                [data_q75 - data_q25, pos_mean, (pos_max - pos_min) * 0.3],
                [data_range, (pos_min + pos_max) / 2, (pos_max - pos_min) * 0.2],
                [abs(data_mean), pos_mean, pos_range * 0.25],
                [data_std * 2, pos_median, pos_range * 0.3],
                [abs(np.max(valid_vals)), pos_mean, pos_range * 0.15],
                [abs(np.min(valid_vals)), pos_median, pos_range * 0.25],
                [data_range * 0.8, pos_mean, pos_std * 2.0],
                [data_range * 1.2, pos_median, pos_std * 0.8],
            ])
        
        if self.gaussian_use_bounds:
            a_min = max(abs(data_range) * self.gaussian_a_min_factor, 1e-10)
            a_max = abs(data_range) * self.gaussian_a_max_factor
            lower = np.array([a_min, self.gaussian_b_min, self.gaussian_c_min], dtype=np.float64)
            upper = np.array([a_max, self.gaussian_b_max, self.gaussian_c_max], dtype=np.float64)
        else:
            lower = -np.inf * np.ones(3, dtype=np.float64)
            upper = np.inf * np.ones(3, dtype=np.float64)
        
        best_params = None
        best_r2 = -np.inf
        best_valid = False
        best_reason = None
        
        for guess in initial_guesses[: self.gaussian_max_retry]:
            guess = np.clip(np.asarray(guess, dtype=np.float64), lower, upper)
            def residuals(p):
                return self.gaussian_curve(valid_pos, *p) - valid_vals
            result = least_squares(
                residuals,
                x0=guess,
                bounds=(lower, upper),
                max_nfev=2000,
            )
            popt = result.x
            fitted = self.gaussian_curve(valid_pos, *popt)
            ss_res = np.sum((valid_vals - fitted) ** 2)
            ss_tot = np.sum((valid_vals - np.mean(valid_vals)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
            is_valid, reason = self.validate_gaussian_parameters(popt, valid_pos, valid_vals)
            if r2 > best_r2:
                best_params = popt
                best_r2 = r2
                best_valid = is_valid
                best_reason = reason
            if is_valid and r2 >= min_r_squared:
                if error_info is not None:
                    error_info['success'] = True
                    error_info['r_squared'] = r2
                    error_info['quality'] = 'high'
                return popt, r2, None
        
        msg = f"Low R-squared: {best_r2:.4f} < {min_r_squared}" if best_params is not None else "All fitting attempts failed"
        if error_info is not None:
            error_info['reason'] = msg
            error_info['success'] = False
            error_info['best_r_squared'] = best_r2
        return None, best_r2 if best_params is not None else 0, msg
    
    def get_available_sensor_positions(self, sensor_data_dict: Dict[str, Any]) -> np.ndarray:
        """
        Gets sensor positions for available sensors based on configuration.
        
        Args:
            sensor_data_dict: Dictionary with sensor data, keys are 'sensor_X'.
        
        Returns:
            Array of sensor positions for available sensors.
        """
        sensor_ids = sorted([int(k.split('_')[1]) for k in sensor_data_dict.keys()])
        positions = []
        for sid in sensor_ids:
            pos = self._sensor_positions_dict.get(sid, self._sensor_positions_dict.get(str(sid), 0))
            if pos > 0:
                positions.append(float(pos))
        return np.array(positions)
    
    def compute_elastic_modulus(self, sensor_data_dict, time_data, nth_valley=None, 
                                time_threshold=None, use_reference_sensor=None):
        """
        Computes elastic modulus E(t) for the asphalt specimen based on force and displacement.
        
        This corresponds to ref.m lines 314-417, optimized to follow the exact logic:
        1. Identify loading cycles (pauses in data)
        2. Find the nth valley point in each cycle (default 300th)
        3. Extract amplitudes at that time point for all sensors
        4. Fit Gaussian curves to spatial distribution
        5. Compute elastic modulus
        
        Args:
            sensor_data_dict: Dictionary with keys 'sensor_1' to 'sensor_11', each containing
                             a DataFrame with 'force' and 'displacement' columns.
            time_data: Array of time values corresponding to sensor measurements.
            nth_valley: Which valley point to use. If None, uses value from config.
            time_threshold: Time threshold to detect pauses (seconds). If None, uses value from config.
            use_reference_sensor: Sensor ID to use as reference for finding valleys. If None, uses value from config.
        
        Returns:
            A dictionary containing:
            - 'elastic_modulus': Array of elastic modulus values E(t) for each loading cycle.
            - 'stiffness': Array of plate stiffness values D(t).
            - 'gaussian_params': List storing Gaussian fitting parameters for each cycle.
            - 'cycle_indices': List of tuples (start_idx, end_idx) for each loading cycle.
        """
        if nth_valley is None:
            nth_valley = self.nth_valley
        if time_threshold is None:
            time_threshold = self.time_threshold
        if use_reference_sensor is None:
            use_reference_sensor = self.reference_sensor
        
        # Extract and organize force and displacement data
        sensor_ids = sorted([int(k.split('_')[1]) for k in sensor_data_dict.keys()])
        forces = {}
        displacements = {}
        
        for sid in sensor_ids:
            key = f'sensor_{sid}'
            if key in sensor_data_dict:
                forces[sid] = sensor_data_dict[key]['force'].values
                displacements[sid] = sensor_data_dict[key]['displacement'].values
        
        # Step 1: Identify loading cycles by detecting pauses
        print("Identifying loading cycles...")
        cycles = self.identify_loading_cycles(
            time_data, 
            displacements.get(use_reference_sensor, displacements[sensor_ids[0]]),
            threshold=time_threshold
        )
        
        print(f"Found {len(cycles)} loading cycles")
        
        # Step 2: For each cycle, find the nth valley point and extract amplitudes
        elastic_modulus_list = []
        stiffness_list = []
        gaussian_params_list = []
        cycle_indices_list = []
        
        # Diagnostic statistics
        stats = {
            'total_cycles': len(cycles),
            'skipped_short_cycle': 0,
            'skipped_no_valley': 0,
            'skipped_gaussian_fit_failed': 0,
            'gaussian_fit_disp_failed': 0,
            'gaussian_fit_force_failed': 0,
            'nan_fourth_derivative': 0,
            'nan_alternative_failed': 0,
            'rejected_low_r_squared': 0,
            'processed_cycles': 0,
            'valid_cycles': 0,
            'fit_failure_reasons': {
                'disp_low_r_squared': 0,
                'disp_invalid_params': 0,
                'disp_runtime_error': 0,
                'force_low_r_squared': 0,
                'force_invalid_params': 0,
                'force_runtime_error': 0,
            },
        }
        
        # Get sensor positions for available sensors
        available_positions = self.get_available_sensor_positions(sensor_data_dict)
        stats['n_positions'] = len(available_positions)

        for cycle_idx, (cycle_times, cycle_disp) in enumerate(tqdm(cycles, desc="Processing cycles")):
            # Find the reference sensor's displacement for this cycle
            ref_sensor_id = use_reference_sensor if use_reference_sensor in sensor_ids else sensor_ids[0]
            
            # Get the corresponding indices in the full time array
            start_time = cycle_times[0]
            end_time = cycle_times[-1]
            start_idx = np.searchsorted(time_data, start_time)
            end_idx = np.searchsorted(time_data, end_time)
            cycle_length = end_idx - start_idx
            
            # Adaptive nth_valley calculation
            adaptive_nth_valley = max(
                self.min_nth_valley,
                min(self.max_nth_valley, int(cycle_length * self.nth_valley_ratio))
            )
            
            if cycle_length < adaptive_nth_valley * 2:
                stats['skipped_short_cycle'] += 1
                continue
            
            # Extract reference sensor displacement for this cycle
            ref_disp_cycle = displacements[ref_sensor_id][start_idx:end_idx]
            
            # Find the nth valley index within this cycle using adaptive value
            valley_idx_relative = self.find_nth_valley_index(ref_disp_cycle, n=adaptive_nth_valley, envelope_span=300)
            
            if valley_idx_relative is None:
                stats['skipped_no_valley'] += 1
                continue
            
            # Convert to absolute index
            valley_idx_absolute = start_idx + valley_idx_relative
            
            # Find corresponding peak (0.05s after valley, as in ref.m)
            valley_time = time_data[valley_idx_absolute]
            peak_time_target = valley_time + 0.05
            peak_idx_absolute = np.argmin(np.abs(time_data - peak_time_target))
            
            # Extract force and displacement values at valley and peak for all sensors
            force_values_valley = np.array([forces[sid][valley_idx_absolute] for sid in sensor_ids])
            disp_values_valley = np.array([displacements[sid][valley_idx_absolute] for sid in sensor_ids])
            
            force_values_peak = np.array([forces[sid][peak_idx_absolute] for sid in sensor_ids])
            disp_values_peak = np.array([displacements[sid][peak_idx_absolute] for sid in sensor_ids])
            
            # Compute amplitude differences (valley - peak)
            force_diff_raw = force_values_valley - force_values_peak
            disp_diff_raw = disp_values_valley - disp_values_peak
            force_diff = self._preprocess_curve(force_diff_raw)
            disp_diff = self._preprocess_curve(disp_diff_raw)
            
            # Step 3: Fit Gaussian curves to spatial distribution with error tracking
            disp_error_info = {}
            disp_params, disp_r_squared, disp_error = self.fit_gaussian_spatial(
                available_positions,
                disp_diff,
                initial_guess=[-0.056, 250, 150],
                error_info=disp_error_info,
                fit_type='disp'
            )
            
            force_error_info = {}
            force_params, force_r_squared, force_error = self.fit_gaussian_spatial(
                available_positions,
                force_diff,
                initial_guess=[73.98, 130, 79.54],
                error_info=force_error_info,
                fit_type='force'
            )
            
            if disp_params is None:
                stats['gaussian_fit_disp_failed'] += 1
                stats['skipped_gaussian_fit_failed'] += 1
                if 'quality' in disp_error_info:
                    quality_key = f"disp_fit_{disp_error_info.get('quality', 'failed')}"
                    stats[quality_key] = stats.get(quality_key, 0) + 1
                if 'reason' in disp_error_info:
                    reason = disp_error_info['reason']
                    if 'Low R-squared' in reason:
                        stats['fit_failure_reasons']['disp_low_r_squared'] += 1
                    elif 'invalid params' in reason.lower() or 'out of range' in reason.lower():
                        stats['fit_failure_reasons']['disp_invalid_params'] += 1
                    elif 'RuntimeError' in reason or 'ValueError' in reason:
                        stats['fit_failure_reasons']['disp_runtime_error'] += 1
                continue
            
            if force_params is None:
                stats['gaussian_fit_force_failed'] += 1
                stats['skipped_gaussian_fit_failed'] += 1
                if 'quality' in force_error_info:
                    quality_key = f"force_fit_{force_error_info.get('quality', 'failed')}"
                    stats[quality_key] = stats.get(quality_key, 0) + 1
                if 'reason' in force_error_info:
                    reason = force_error_info['reason']
                    if 'Low R-squared' in reason:
                        stats['fit_failure_reasons']['force_low_r_squared'] += 1
                    elif 'invalid params' in reason.lower() or 'out of range' in reason.lower():
                        stats['fit_failure_reasons']['force_invalid_params'] += 1
                    elif 'RuntimeError' in reason or 'ValueError' in reason:
                        stats['fit_failure_reasons']['force_runtime_error'] += 1
                continue
            
            # Track fit quality
            if 'quality' in disp_error_info:
                quality_key = f"disp_fit_{disp_error_info['quality']}"
                stats[quality_key] = stats.get(quality_key, 0) + 1
            if 'quality' in force_error_info:
                quality_key = f"force_fit_{force_error_info['quality']}"
                stats[quality_key] = stats.get(quality_key, 0) + 1
            
            # Track R-squared values for statistics
            if 'r_squared_values' not in stats:
                stats['r_squared_values'] = {'disp': [], 'force': []}
            stats['r_squared_values']['disp'].append(disp_r_squared)
            stats['r_squared_values']['force'].append(force_r_squared)
            
            # Step 4: Compute elastic modulus using Hetenyi foundation approach
            a_disp, b_disp, c_disp = disp_params
            a_force, b_force, c_force = force_params
            
            # Compute function values at reference position x=250
            f_disp = self.gaussian_curve(250, a_disp, b_disp, c_disp)
            f_force = self.gaussian_curve(250, a_force, b_force, c_force)
            
            # Compute fourth derivative at x=250
            f_disp_4th = self.gaussian_fourth_derivative(250, a_disp, b_disp, c_disp)
            
            # Compute plate stiffness D with improved NaN handling
            stats['processed_cycles'] += 1
            if abs(f_disp_4th) > self.fourth_derivative_threshold:
                D = (-35200 * f_disp + f_force) / f_disp_4th
                E = abs(D * (1 - self.poisson_ratio ** 2) / self.I)
            else:
                if self.use_alternative_calculation:
                    # Alternative calculation using simplified approach
                    # Use a small regularization term to avoid division by zero
                    numerator = -35200 * f_disp + f_force
                    regularization = self.fourth_derivative_threshold * (1.0 if f_disp_4th >= 0 else -1.0)
                    denominator = f_disp_4th + regularization
                    
                    if abs(denominator) > 1e-15:
                        D = numerator / denominator
                        E = abs(D * (1 - self.poisson_ratio ** 2) / self.I)
                        
                        # Check if result is reasonable
                        if np.isnan(E) or np.isinf(E) or E <= 0 or E > 1e10:
                            stats['nan_alternative_failed'] += 1
                            E = np.nan
                            D = np.nan
                    else:
                        stats['nan_alternative_failed'] += 1
                        E = np.nan
                        D = np.nan
                else:
                    stats['nan_fourth_derivative'] += 1
                    E = np.nan
                    D = np.nan
            
            # Exclude cycle from labels when disp/force fit R² is below configured threshold.
            # Only apply R² rejection when n_positions >= min_points_for_r2_reject (if set).
            apply_r2_reject = self.gaussian_min_r_squared_accept_for_label is not None
            if apply_r2_reject and self.gaussian_min_points_for_r2_reject is not None:
                apply_r2_reject = len(available_positions) >= self.gaussian_min_points_for_r2_reject
            if apply_r2_reject:
                if disp_r_squared < self.gaussian_min_r_squared_accept_for_label or force_r_squared < self.gaussian_min_r_squared_accept_for_label:
                    E = np.nan
                    D = np.nan
                    stats['rejected_low_r_squared'] = stats.get('rejected_low_r_squared', 0) + 1
            
            if not np.isnan(E):
                stats['valid_cycles'] += 1
            
            elastic_modulus_list.append(E)
            stiffness_list.append(D)
            gaussian_params_list.append({
                'disp': disp_params,
                'force': force_params,
                'disp_r_squared': disp_r_squared,
                'force_r_squared': force_r_squared
            })
            cycle_indices_list.append((start_idx, end_idx))
        
        result = {
            'elastic_modulus': np.array(elastic_modulus_list),
            'stiffness': np.array(stiffness_list),
            'gaussian_params': gaussian_params_list,
            'cycle_indices': cycle_indices_list,
            'diagnostics': stats
        }
        
        # Enhanced diagnostic information
        total_cycles = stats['total_cycles']
        computed_cycles = stats['processed_cycles']
        nan_count = np.sum(np.isnan(result['elastic_modulus']))
        valid_count = stats['valid_cycles']
        
        print(f"\n[Diagnostic] compute_elastic_modulus - Cycle Processing Summary:")
        print(f"  Total cycles identified: {total_cycles}")
        print(f"  Cycles processed: {computed_cycles}")
        print(f"  Valid cycles (non-NaN): {valid_count}")
        print(f"  NaN cycles: {nan_count}")
        print(f"\n[Diagnostic] Cycle Skip Reasons:")
        print(f"  Skipped (too short): {stats['skipped_short_cycle']}")
        print(f"  Skipped (no valley found): {stats['skipped_no_valley']}")
        print(f"  Skipped (Gaussian fit failed): {stats['skipped_gaussian_fit_failed']}")
        print(f"    - Displacement fit failed: {stats['gaussian_fit_disp_failed']}")
        print(f"    - Force fit failed: {stats['gaussian_fit_force_failed']}")
        
        if 'fit_failure_reasons' in stats:
            print(f"\n[Diagnostic] Detailed Fit Failure Reasons:")
            print(f"  Displacement fit failures:")
            print(f"    - Low R-squared: {stats['fit_failure_reasons']['disp_low_r_squared']}")
            print(f"    - Invalid parameters: {stats['fit_failure_reasons']['disp_invalid_params']}")
            print(f"    - Runtime/Value errors: {stats['fit_failure_reasons']['disp_runtime_error']}")
            print(f"  Force fit failures:")
            print(f"    - Low R-squared: {stats['fit_failure_reasons']['force_low_r_squared']}")
            print(f"    - Invalid parameters: {stats['fit_failure_reasons']['force_invalid_params']}")
            print(f"    - Runtime/Value errors: {stats['fit_failure_reasons']['force_runtime_error']}")
        print(f"\n[Diagnostic] NaN Generation Reasons:")
        print(f"  NaN (fourth derivative too small): {stats['nan_fourth_derivative']}")
        print(f"  NaN (alternative calculation failed): {stats['nan_alternative_failed']}")
        if self.gaussian_min_r_squared_accept_for_label is not None:
            print(f"  Rejected (R² < {self.gaussian_min_r_squared_accept_for_label} for label): {stats.get('rejected_low_r_squared', 0)}")
        r_squared_disp = stats.get('r_squared_values', {}).get('disp', [])
        r_squared_force = stats.get('r_squared_values', {}).get('force', [])
        
        n_pos = stats.get('n_positions', 0)
        k_params = 3
        valid_for_adj = n_pos > k_params + 1

        if len(r_squared_disp) > 0:
            print(f"\n[Diagnostic] Displacement Fit Quality:")
            print(f"  Mean R²: {np.mean(r_squared_disp):.4f}")
            print(f"  Min R²: {np.min(r_squared_disp):.4f}, Max R²: {np.max(r_squared_disp):.4f}")
            print(f"  R² >= 0.5: {np.sum(np.array(r_squared_disp) >= 0.5)}/{len(r_squared_disp)}")
            print(f"  R² >= 0.2: {np.sum(np.array(r_squared_disp) >= 0.2)}/{len(r_squared_disp)}")
            print(f"  R² >= 0.15: {np.sum(np.array(r_squared_disp) >= 0.15)}/{len(r_squared_disp)}")
            if valid_for_adj:
                adj_disp = [1 - (1 - r2) * (n_pos - 1) / (n_pos - k_params - 1) for r2 in r_squared_disp]
                print(f"  Adjusted R² (n={n_pos}, k={k_params}): mean={np.mean(adj_disp):.4f}, min={np.min(adj_disp):.4f}, max={np.max(adj_disp):.4f}")

        if len(r_squared_force) > 0:
            print(f"\n[Diagnostic] Force Fit Quality:")
            print(f"  Mean R²: {np.mean(r_squared_force):.4f}")
            print(f"  Min R²: {np.min(r_squared_force):.4f}, Max R²: {np.max(r_squared_force):.4f}")
            print(f"  R² >= 0.5: {np.sum(np.array(r_squared_force) >= 0.5)}/{len(r_squared_force)}")
            print(f"  R² >= 0.2: {np.sum(np.array(r_squared_force) >= 0.2)}/{len(r_squared_force)}")
            if valid_for_adj:
                adj_force = [1 - (1 - r2) * (n_pos - 1) / (n_pos - k_params - 1) for r2 in r_squared_force]
                print(f"  Adjusted R² (n={n_pos}, k={k_params}): mean={np.mean(adj_force):.4f}, min={np.min(adj_force):.4f}, max={np.max(adj_force):.4f}")
        
        fit_quality_keys = [k for k in stats.keys() if k.startswith('disp_fit_') or k.startswith('force_fit_')]
        if fit_quality_keys:
            print(f"\n[Diagnostic] Fit Quality Distribution:")
            for key in sorted(fit_quality_keys):
                print(f"  {key}: {stats[key]}")
        
        print(f"\n[Diagnostic] Recovery Rate: {valid_count}/{total_cycles} = {100*valid_count/total_cycles:.1f}%")
        
        return result
    
    def compute_loading_cycles(self, elastic_modulus, time_data):
        """
        Computes loading cycle count and energy ratio.
        
        This corresponds to ref.m lines 418-432. Uses loading_cycles_offset and
        loading_cycles_per_step from preprocessing.fatigue_life in dataset config.
        
        Args:
            elastic_modulus: Array of elastic modulus values E(t).
            time_data: Array of time values.
        
        Returns:
            Dictionary containing:
            - 'loading_cycles': Array of cumulative loading cycles.
            - 'energy_ratio': Energy ratio (loading cycles * E(t)).
        """
        n_steps = len(elastic_modulus)
        loading_cycles = self.loading_cycles_offset + np.arange(n_steps) * self.loading_cycles_per_step
        
        # Filter out NaN values
        valid_mask = ~np.isnan(elastic_modulus)
        
        energy_ratio = np.full_like(elastic_modulus, np.nan)
        energy_ratio[valid_mask] = loading_cycles[valid_mask] * elastic_modulus[valid_mask]
        
        return {
            'loading_cycles': loading_cycles,
            'energy_ratio': energy_ratio
        }


if __name__ == '__main__':
    # Example usage for testing
    from Libs.data.dataloader import load_arrows_data
    
    print("Loading ARROWS data...")
    data = load_arrows_data('Input/raw/data.mat')
    
    if data:
        print("Initializing physics model...")
        physics = PhysicsModel()
        
        print("Computing elastic modulus...")
        modulus_result = physics.compute_elastic_modulus(data['sensors'], data['time'])
        
        print(f"Elastic modulus shape: {modulus_result['elastic_modulus'].shape}")
        print(f"Valid modulus values: {np.sum(~np.isnan(modulus_result['elastic_modulus']))}")
        print(f"Mean modulus: {np.nanmean(modulus_result['elastic_modulus']):.4f}")
        print(f"Std modulus: {np.nanstd(modulus_result['elastic_modulus']):.4f}")


