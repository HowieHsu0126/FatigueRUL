from typing import Any, Dict, List, Tuple, Optional

import os

import numpy as np
import pandas as pd
import scipy.io

from Libs.config import load_dataset_config
from Libs.data.quality_checker import SensorQualityChecker


def interpolate_displacement(
    disp_source_1: np.ndarray,
    disp_source_2: np.ndarray,
    pos_source_1: float,
    pos_target: float,
    pos_source_2: float
) -> np.ndarray:
    """
    Interpolates displacement based on two source sensors using distance-weighted interpolation.
    
    Uses distance-weighted interpolation: disp_target = (disp_1 * w_1 + disp_2 * w_2)
    where weights are inversely proportional to distance.
    
    Args:
        disp_source_1: Displacement data from first source sensor.
        disp_source_2: Displacement data from second source sensor.
        pos_source_1: Position of first source sensor (mm).
        pos_target: Position of target sensor (mm).
        pos_source_2: Position of second source sensor (mm).
    
    Returns:
        Interpolated displacement array for target sensor.
    """
    d_1_target = abs(pos_target - pos_source_1)
    d_target_2 = abs(pos_source_2 - pos_target)
    
    w_1 = d_target_2 / (d_1_target + d_target_2)
    w_2 = d_1_target / (d_1_target + d_target_2)
    
    disp_target = disp_source_1 * w_1 + disp_source_2 * w_2
    return disp_target


def load_arrows_data(mat_file_path: str) -> Dict[str, Any]:
    """
    Loads data from the ARROWS .mat file and processes it according to the logic
    found in the ref.m script.

    Args:
        mat_file_path: The path to the .mat data file.

    Returns:
        A dictionary containing the processed data, including:
        'time': A numpy array of time values.
        'temp_ambient': A numpy array for ambient temperature.
        'temp_underside': A numpy array for underside temperature.
        'sensors': A dictionary where keys are sensor IDs (e.g., 'sensor_1')
                   and values are pandas DataFrames with 'force' and 'displacement'
                   columns.
        'file_header': The raw file header information.
    """
    if not os.path.exists(mat_file_path):
        raise FileNotFoundError(f"The file {mat_file_path} was not found.")
    mat = scipy.io.loadmat(mat_file_path)

    config = load_dataset_config()
    sensors_config = config.get('sensors', {})
    channel_mapping = sensors_config.get('channel_mapping', {})
    sensor_10_interpolation = sensors_config.get('sensor_10_interpolation', {})
    sensor_positions = sensors_config.get('sensor_positions', {})
    
    channel_map = {}
    for sensor_id, channels in channel_mapping.items():
        sensor_id_int = int(sensor_id) if isinstance(sensor_id, str) else sensor_id
        force_ch = channels[0]
        disp_ch = channels[1] if len(channels) > 1 and channels[1] is not None else None
        channel_map[sensor_id_int] = (force_ch, disp_ch)

    processed_data = {
        'sensors': {}
    }

    # Extract time and temperature data, flattening them to 1D arrays
    # Using Channel_1 for time as the primary time reference from ref.m
    processed_data['time'] = mat.get('Channel_1_Data', np.array([])).flatten()
    processed_data['temp_ambient'] = mat.get('Channel_17_Data', np.array([])).flatten()
    processed_data['temp_underside'] = mat.get('Channel_18_Data', np.array([])).flatten()

    # First pass: process all sensors with complete force-displacement pairs
    for sensor_id, (force_ch, disp_ch) in channel_map.items():
        if disp_ch is None:
            continue
        
        force_key = f'Channel_{force_ch}_Data'
        disp_key = f'Channel_{disp_ch}_Data'
        
        if force_key not in mat or disp_key not in mat:
            print(f"Warning: Data channels not found for Sensor {sensor_id}. Skipping.")
            continue
        
        force_data = mat[force_key].flatten()
        disp_data = mat[disp_key].flatten()
        
        if len(force_data) != len(processed_data['time']) or len(disp_data) != len(processed_data['time']):
            print(f"Warning: Data length mismatch for Sensor {sensor_id}. Skipping.")
            continue
        
        sensor_df = pd.DataFrame({
            'force': force_data,
            'displacement': disp_data
        })
        processed_data['sensors'][f'sensor_{sensor_id}'] = sensor_df
    
    # Second pass: process sensors with interpolated displacement
    for sensor_id, (force_ch, disp_ch) in channel_map.items():
        if disp_ch is None:
            force_key = f'Channel_{force_ch}_Data'
            
            if force_key not in mat:
                print(f"Warning: Force channel not found for Sensor {sensor_id}. Skipping.")
                continue
            elif len(mat[force_key].flatten()) != len(processed_data['time']):
                print(f"Warning: Force data length mismatch for Sensor {sensor_id}. Skipping.")
                continue
            
            required_sensors = sensor_10_interpolation.get('required_sensors', [])
            interpolation_positions = sensor_10_interpolation.get('interpolation_positions', {})
            
            sensor_id_str = str(sensor_id)
            if sensor_id_str in interpolation_positions or sensor_id in interpolation_positions:
                target_pos_key = sensor_id_str if sensor_id_str in interpolation_positions else sensor_id
                target_pos = float(interpolation_positions[target_pos_key])
                source_sensor_ids = [int(sid) if isinstance(sid, str) else sid for sid in required_sensors]
                source_sensor_keys = [f'sensor_{sid}' for sid in source_sensor_ids]
                
                available_sources = [key for key in source_sensor_keys if key in processed_data['sensors']]
                
                if len(available_sources) == len(source_sensor_keys):
                    force_data = mat[force_key].flatten()
                    source_1_id = source_sensor_ids[0]
                    source_2_id = source_sensor_ids[1]
                    source_1_key = f'sensor_{source_1_id}'
                    source_2_key = f'sensor_{source_2_id}'
                    
                    disp_source_1 = processed_data['sensors'][source_1_key]['displacement'].values
                    disp_source_2 = processed_data['sensors'][source_2_key]['displacement'].values
                    pos_source_1_key = str(source_1_id) if str(source_1_id) in interpolation_positions else source_1_id
                    pos_source_2_key = str(source_2_id) if str(source_2_id) in interpolation_positions else source_2_id
                    pos_source_1 = float(interpolation_positions[pos_source_1_key])
                    pos_source_2 = float(interpolation_positions[pos_source_2_key])
                    
                    disp_data = interpolate_displacement(
                        disp_source_1, disp_source_2,
                        pos_source_1, target_pos, pos_source_2
                    )
                    
                    sensor_df = pd.DataFrame({
                        'force': force_data,
                        'displacement': disp_data
                    })
                    processed_data['sensors'][f'sensor_{sensor_id}'] = sensor_df
                    print(f"Info: Interpolated displacement for Sensor {sensor_id} from sensors {source_1_id} and {source_2_id}.")
                else:
                    missing = [sid for sid, key in zip(source_sensor_ids, source_sensor_keys) if key not in available_sources]
                    print(f"Warning: Cannot interpolate displacement for Sensor {sensor_id}. Required sensors {missing} not available.")
    
    # Store header info for metadata reference
    processed_data['file_header'] = mat.get('File_Header')

    print(f"Successfully loaded data for {len(processed_data['sensors'])} sensors.")
    
    # Perform quality check and filter abnormal sensors
    config = load_dataset_config()
    quality_config = config.get('preprocessing', {}).get('quality_check', {})
    
    if quality_config.get('enable', True):
        print("\n" + "=" * 60)
        print("Performing sensor data quality check...")
        print("=" * 60)
        
        checker = SensorQualityChecker(
            force_max_threshold=float(quality_config.get('force_max_threshold', 500000.0)),
            displacement_max_threshold=float(quality_config.get('displacement_max_threshold', 1e15)),
            jump_ratio_threshold=float(quality_config.get('jump_ratio_threshold', 10.0)),
            stability_window_ratio=float(quality_config.get('stability_window_ratio', 0.3)),
            stability_variance_ratio_threshold=float(quality_config.get('stability_variance_ratio_threshold', 10.0))
        )
        
        filtered_sensors, removed_sensors = checker.filter_sensors(processed_data['sensors'])
        
        # If sensor_9 was removed, sensor_10's displacement (interpolated from 9 and 11) is inconsistent.
        # Re-interpolate from 8 and 11 when both exist; otherwise remove sensor_10.
        sensors_config = config.get('sensors', {})
        sensor_10_interpolation = sensors_config.get('sensor_10_interpolation', {})
        interpolation_positions = sensor_10_interpolation.get('interpolation_positions', {})
        sensor_positions = sensors_config.get('sensor_positions', {})
        if 'sensor_9' in removed_sensors and 'sensor_10' in filtered_sensors:
            if 'sensor_8' in filtered_sensors and 'sensor_11' in filtered_sensors:
                pos_8 = float(sensor_positions.get(8, sensor_positions.get('8', 320)))
                pos_10 = float(interpolation_positions.get(10, interpolation_positions.get('10', 390)))
                pos_11 = float(interpolation_positions.get(11, interpolation_positions.get('11', 425)))
                disp_8 = filtered_sensors['sensor_8']['displacement'].values
                disp_11 = filtered_sensors['sensor_11']['displacement'].values
                new_disp = interpolate_displacement(disp_8, disp_11, pos_8, pos_10, pos_11)
                filtered_sensors['sensor_10'] = pd.DataFrame({
                    'force': filtered_sensors['sensor_10']['force'].values,
                    'displacement': new_disp
                })
                print("Info: Re-interpolated displacement for Sensor 10 from sensors 8 and 11 (sensor 9 was removed).")
            else:
                del filtered_sensors['sensor_10']
                print("Info: Removed sensor_10 because its interpolation source sensor_9 was removed and alternative (sensor_8, sensor_11) is not both available.")
        
        if removed_sensors:
            print(f"\nRemoved {len(removed_sensors)} abnormal sensor(s):")
            for sensor_id, reasons in removed_sensors.items():
                print(f"  - {sensor_id}:")
                for reason in reasons:
                    print(f"    * {reason}")
        else:
            print("\nAll sensors passed quality check.")
        
        processed_data['sensors'] = filtered_sensors
        print(f"\nRemaining sensors after quality check: {len(processed_data['sensors'])}")
    
    return processed_data

if __name__ == '__main__':
    # Example usage:
    data_path = 'Input/raw/data.mat'
    arrows_data = load_arrows_data(data_path)

    if arrows_data:
        print("\nData loading summary:")
        print(f"Time points loaded: {len(arrows_data['time'])}")
        print(f"Ambient temperature points loaded: {len(arrows_data['temp_ambient'])}")
        print(f"Loaded sensors: {list(arrows_data['sensors'].keys())}")

        # Print a sample of the data for the first available sensor
        if arrows_data['sensors']:
            first_sensor_key = list(arrows_data['sensors'].keys())[0]
            print(f"\nSample data from '{first_sensor_key}':")
            print(arrows_data['sensors'][first_sensor_key].head())
