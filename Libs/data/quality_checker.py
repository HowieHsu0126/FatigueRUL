"""
Sensor data quality checker for detecting and filtering abnormal sensor measurements.

This module implements quality checks for force and displacement sensor data,
including extreme value detection, sudden jump detection, and stability analysis.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class SensorQualityChecker:
    """
    Checks sensor data quality and identifies abnormal sensors.
    
    Detects three types of abnormalities:
    1. Extreme values: Data exceeding reasonable thresholds
    2. Sudden jumps: Abrupt changes to extreme values
    3. Instability: Data that is stable initially but becomes unstable later
    """
    
    def __init__(
        self,
        force_max_threshold: float = 500000.0,
        displacement_max_threshold: float = 1e15,
        jump_ratio_threshold: float = 10.0,
        stability_window_ratio: float = 0.3,
        stability_variance_ratio_threshold: float = 10.0
    ):
        """
        Initialize the quality checker with detection thresholds.
        
        Args:
            force_max_threshold: Maximum acceptable force value in Newtons.
            displacement_max_threshold: Maximum acceptable displacement value in millimeters.
            jump_ratio_threshold: Threshold for detecting sudden jumps (ratio of change).
            stability_window_ratio: Ratio of data length to use for front/back stability analysis.
            stability_variance_ratio_threshold: Threshold for variance ratio (back/front) to detect instability.
        """
        self.force_max_threshold = float(force_max_threshold)
        self.displacement_max_threshold = float(displacement_max_threshold)
        self.jump_ratio_threshold = float(jump_ratio_threshold)
        self.stability_window_ratio = float(stability_window_ratio)
        self.stability_variance_ratio_threshold = float(stability_variance_ratio_threshold)
    
    def detect_extreme_values(
        self,
        force_data: np.ndarray,
        displacement_data: np.ndarray
    ) -> Tuple[bool, List[str]]:
        """
        Detects extreme values exceeding thresholds.
        
        Args:
            force_data: Array of force measurements.
            displacement_data: Array of displacement measurements.
        
        Returns:
            Tuple of (is_abnormal, reasons) where reasons is a list of detected issues.
        """
        reasons = []
        is_abnormal = False
        
        force_data_numeric = pd.to_numeric(force_data, errors='coerce')
        displacement_data_numeric = pd.to_numeric(displacement_data, errors='coerce')
        
        force_max = np.nanmax(np.abs(force_data_numeric))
        disp_max = np.nanmax(np.abs(displacement_data_numeric))
        
        if not np.isnan(force_max) and force_max > self.force_max_threshold:
            reasons.append(f"Extreme force value detected: {force_max:.2e} N (threshold: {self.force_max_threshold:.2e} N)")
            is_abnormal = True
        
        if not np.isnan(disp_max) and disp_max > self.displacement_max_threshold:
            reasons.append(f"Extreme displacement value detected: {disp_max:.2e} mm (threshold: {self.displacement_max_threshold:.2e} mm)")
            is_abnormal = True
        
        return is_abnormal, reasons
    
    def detect_sudden_jumps(
        self,
        force_data: np.ndarray,
        displacement_data: np.ndarray
    ) -> Tuple[bool, List[str]]:
        """
        Detects sudden jumps to extreme values.
        
        Args:
            force_data: Array of force measurements.
            displacement_data: Array of displacement measurements.
        
        Returns:
            Tuple of (is_abnormal, reasons) where reasons is a list of detected issues.
        """
        reasons = []
        is_abnormal = False
        
        force_data_numeric = pd.to_numeric(force_data, errors='coerce')
        displacement_data_numeric = pd.to_numeric(displacement_data, errors='coerce')
        
        force_abs = np.abs(force_data_numeric)
        disp_abs = np.abs(displacement_data_numeric)
        
        force_diff = np.diff(force_abs)
        disp_diff = np.diff(disp_abs)
        
        force_prev = force_abs[:-1]
        disp_prev = disp_abs[:-1]
        
        valid_force_mask = force_prev > 1e-10
        valid_disp_mask = disp_prev > 1e-10
        
        force_change_ratio = np.zeros_like(force_diff, dtype=np.float64)
        disp_change_ratio = np.zeros_like(disp_diff, dtype=np.float64)
        
        force_change_ratio[valid_force_mask] = np.abs(force_diff[valid_force_mask]) / force_prev[valid_force_mask]
        disp_change_ratio[valid_disp_mask] = np.abs(disp_diff[valid_disp_mask]) / disp_prev[valid_disp_mask]
        
        force_jump_mask = (force_change_ratio > self.jump_ratio_threshold) & (force_abs[1:] > self.force_max_threshold * 0.8)
        disp_jump_mask = (disp_change_ratio > self.jump_ratio_threshold) & (disp_abs[1:] > self.displacement_max_threshold * 0.8)
        
        if np.any(force_jump_mask):
            jump_idx = np.where(force_jump_mask)[0][0]
            jump_value = force_abs[jump_idx + 1]
            reasons.append(f"Sudden force jump detected at index {jump_idx}: {jump_value:.2e} N")
            is_abnormal = True
        
        if np.any(disp_jump_mask):
            jump_idx = np.where(disp_jump_mask)[0][0]
            jump_value = disp_abs[jump_idx + 1]
            reasons.append(f"Sudden displacement jump detected at index {jump_idx}: {jump_value:.2e} mm")
            is_abnormal = True
        
        return is_abnormal, reasons
    
    def detect_instability(
        self,
        force_data: np.ndarray,
        displacement_data: np.ndarray
    ) -> Tuple[bool, List[str]]:
        """
        Detects instability where data is stable initially but becomes unstable later.
        
        Args:
            force_data: Array of force measurements.
            displacement_data: Array of displacement measurements.
        
        Returns:
            Tuple of (is_abnormal, reasons) where reasons is a list of detected issues.
        """
        reasons = []
        is_abnormal = False
        
        force_data_numeric = pd.to_numeric(force_data, errors='coerce')
        displacement_data_numeric = pd.to_numeric(displacement_data, errors='coerce')
        
        data_length = len(force_data_numeric)
        window_size = int(data_length * self.stability_window_ratio)
        
        if window_size < 10:
            return False, []
        
        force_abs = np.abs(force_data_numeric)
        disp_abs = np.abs(displacement_data_numeric)
        
        front_force = force_abs[:window_size]
        back_force = force_abs[-window_size:]
        front_disp = disp_abs[:window_size]
        back_disp = disp_abs[-window_size:]
        
        front_force_var = np.nanvar(front_force)
        back_force_var = np.nanvar(back_force)
        front_disp_var = np.nanvar(front_disp)
        back_disp_var = np.nanvar(back_disp)
        
        force_variance_ratio = 0.0
        disp_variance_ratio = 0.0
        
        if front_force_var > 1e-10:
            force_variance_ratio = back_force_var / front_force_var
        elif back_force_var > 1e-10:
            force_variance_ratio = 1e10
        
        if front_disp_var > 1e-10:
            disp_variance_ratio = back_disp_var / front_disp_var
        elif back_disp_var > 1e-10:
            disp_variance_ratio = 1e10
        
        if force_variance_ratio > self.stability_variance_ratio_threshold:
            reasons.append(
                f"Force instability detected: variance ratio = {force_variance_ratio:.2f} "
                f"(front: {front_force_var:.2e}, back: {back_force_var:.2e})"
            )
            is_abnormal = True
        
        if disp_variance_ratio > self.stability_variance_ratio_threshold:
            reasons.append(
                f"Displacement instability detected: variance ratio = {disp_variance_ratio:.2f} "
                f"(front: {front_disp_var:.2e}, back: {back_disp_var:.2e})"
            )
            is_abnormal = True
        
        return is_abnormal, reasons
    
    def check_sensor(
        self,
        sensor_id: str,
        sensor_df: pd.DataFrame
    ) -> Tuple[bool, List[str]]:
        """
        Performs comprehensive quality check on a single sensor.
        
        Args:
            sensor_id: Sensor identifier (e.g., 'sensor_1').
            sensor_df: DataFrame with 'force' and 'displacement' columns.
        
        Returns:
            Tuple of (is_abnormal, reasons) where reasons is a list of all detected issues.
        """
        force_data = sensor_df['force'].values
        displacement_data = sensor_df['displacement'].values
        
        all_reasons = []
        is_abnormal = False
        
        extreme_abnormal, extreme_reasons = self.detect_extreme_values(force_data, displacement_data)
        if extreme_abnormal:
            is_abnormal = True
            all_reasons.extend(extreme_reasons)
        
        jump_abnormal, jump_reasons = self.detect_sudden_jumps(force_data, displacement_data)
        if jump_abnormal:
            is_abnormal = True
            all_reasons.extend(jump_reasons)
        
        instability_abnormal, instability_reasons = self.detect_instability(force_data, displacement_data)
        if instability_abnormal:
            is_abnormal = True
            all_reasons.extend(instability_reasons)
        
        return is_abnormal, all_reasons
    
    def filter_sensors(
        self,
        sensors_dict: Dict[str, pd.DataFrame]
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[str]]]:
        """
        Filters abnormal sensors from the sensor dictionary.
        
        Args:
            sensors_dict: Dictionary mapping sensor IDs to DataFrames.
        
        Returns:
            Tuple of (filtered_sensors, removed_sensors) where:
            - filtered_sensors: Dictionary containing only normal sensors
            - removed_sensors: Dictionary mapping removed sensor IDs to their reasons
        """
        filtered_sensors = {}
        removed_sensors = {}
        
        for sensor_id, sensor_df in sensors_dict.items():
            is_abnormal, reasons = self.check_sensor(sensor_id, sensor_df)
            
            if is_abnormal:
                removed_sensors[sensor_id] = reasons
            else:
                filtered_sensors[sensor_id] = sensor_df
        
        return filtered_sensors, removed_sensors

