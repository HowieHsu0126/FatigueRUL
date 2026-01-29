"""Time series dataset preparation for deep learning models."""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series data from multiple sensors.
    
    Converts sensor data with time windows into sequences suitable for
    LSTM, GRU, Transformer, and other sequence models.
    """

    def __init__(
        self,
        samples: Sequence[np.ndarray],
        targets: Optional[Sequence[float]] = None,
        device: str = "cpu",
        sample_to_window_idx: Optional[Sequence[int]] = None,
        sample_weights: Optional[Sequence[float]] = None,
    ) -> None:
        """
        Initialize time series dataset.
        
        Args:
            samples: List of time series arrays, each of shape [seq_len, input_dim]
            targets: Optional list of target values (RUL predictions)
            device: Target device for inference (data stored on CPU, moved to device in collate function)
            sample_to_window_idx: Optional mapping from sample index to original window index
        """
        self.samples = [np.asarray(sample, dtype=np.float32) for sample in samples]
        self.targets = None
        if targets is not None:
            self.targets = np.asarray(targets, dtype=np.float32)
        self.sample_to_window_idx = (
            np.asarray(sample_to_window_idx, dtype=np.int64) if sample_to_window_idx is not None else None
        )
        self.sample_weights = (
            np.asarray(sample_weights, dtype=np.float32) if sample_weights is not None else None
        )

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Fetch a single sample.
        
        Args:
            index: Sample index
            
        Returns:
            Tuple of (features, target) where features is [seq_len, input_dim] on CPU
        """
        features = torch.as_tensor(self.samples[index], dtype=torch.float32)
        target = None
        if self.targets is not None:
            target = torch.as_tensor(self.targets[index], dtype=torch.float32)
        weight = None
        if self.sample_weights is not None:
            weight = torch.as_tensor(self.sample_weights[index], dtype=torch.float32)
        return features, target, weight


def prepare_timeseries_data(
    sensor_data_dict: Dict[str, pd.DataFrame],
    time_windows: List[Tuple[int, int]],
    labels: Optional[np.ndarray] = None,
    sample_weights: Optional[np.ndarray] = None,
    flatten_sensors: bool = True,
    device: str = "cpu",
    max_sequence_length: Optional[int] = None,
    sequence_length: Optional[int] = None,
    stride: Optional[int] = None,
    use_full_sequence: bool = False,
    augmentation: Optional[Dict[str, Any]] = None,
) -> TimeSeriesDataset:
    """
    Prepare time series data from sensor measurements.
    
    Converts sensor data into time series format suitable for sequence models.
    Each sample is a time window with all sensor measurements.
    
    Args:
        sensor_data_dict: Dictionary mapping sensor keys (e.g., 'sensor_1') to DataFrames
        time_windows: List of (start_idx, end_idx) tuples defining time windows
        labels: Optional array of target values (RUL) for each time window
        flatten_sensors: If True, flatten to [seq_len, num_sensors*2];
                        If False, keep as [num_sensors, seq_len, 2]
        device: Device to store tensors on
        max_sequence_length: Maximum sequence length. If None, no limit. If sequence is longer, uniform sampling is applied.
        
    Returns:
        TimeSeriesDataset object ready for DataLoader
    """
    sensor_ids = sorted([int(key.split("_")[1]) for key in sensor_data_dict.keys()])
    augmentation = augmentation or {}
    
    sensor_lengths = {sid: len(sensor_data_dict[f"sensor_{sid}"]) for sid in sensor_ids}
    min_length = min(sensor_lengths.values())
    
    def _validate_window(window_idx: int, start_idx: int, end_idx: int) -> None:
        if start_idx < 0 or end_idx <= start_idx:
            raise ValueError(
                f"Invalid time window idx={window_idx}: start={start_idx}, end={end_idx}"
            )
        if end_idx > min_length:
            raise ValueError(
                f"Window idx={window_idx} exceeds sensor length: end={end_idx}, min_length={min_length}"
            )
    
    def _downsample(index_list: List[int], max_len: Optional[int]) -> List[int]:
        if max_len is None or len(index_list) <= max_len:
            return index_list
        step = len(index_list) / max_len
        mapped = [int(index_list[0] + i * step) for i in range(max_len)]
        return [min(idx, index_list[-1]) for idx in mapped]
    
    def _build_transforms(cfg: Dict[str, Any]):
        transforms = []
        if not cfg.get("enable", False):
            return lambda arr: arr
        
        if cfg.get("add_noise", False):
            noise_std = float(cfg.get("noise_std", 0.01))
            transforms.append(lambda arr: arr + np.random.normal(0, noise_std, size=arr.shape).astype(np.float32))
        
        if cfg.get("time_warp", False):
            warp_range = cfg.get("warp_factor_range", [0.9, 1.1])
            low, high = float(warp_range[0]), float(warp_range[1])
            
            def _time_warp(arr: np.ndarray) -> np.ndarray:
                factor = np.random.uniform(low, high)
                orig_len, feat_dim = arr.shape
                warped_len = max(2, int(orig_len * factor))
                base_idx = np.linspace(0, orig_len - 1, warped_len)
                warped = np.stack([np.interp(base_idx, np.arange(orig_len), arr[:, i]) for i in range(feat_dim)], axis=1)
                resample_idx = np.linspace(0, warped_len - 1, orig_len)
                resampled = np.stack([np.interp(resample_idx, np.arange(warped_len), warped[:, i]) for i in range(feat_dim)], axis=1)
                return resampled.astype(np.float32)
            
            transforms.append(_time_warp)
        
        def _compose(arr: np.ndarray) -> np.ndarray:
            output = arr
            for fn in transforms:
                output = fn(output)
            return output
        
        return _compose
    
    apply_transforms = _build_transforms(augmentation)
    
    def build_indices(start_idx: int, end_idx: int) -> List[List[int]]:
        total_len = end_idx - start_idx
        if total_len <= 0:
            return []
        if use_full_sequence:
            full = list(range(start_idx, end_idx))
            return [_downsample(full, max_sequence_length)] if max_sequence_length else [full]
        
        target_len = sequence_length if sequence_length is not None else total_len
        window_stride = stride if stride is not None else target_len
        if total_len < target_len:
            full = list(range(start_idx, end_idx))
            return [_downsample(full, max_sequence_length)] if max_sequence_length else [full]
        
        starts = range(start_idx, end_idx - target_len + 1, window_stride)
        indices = []
        for s in starts:
            window = list(range(s, s + target_len))
            indices.append(_downsample(window, max_sequence_length) if max_sequence_length else window)
        return indices
    
    samples = []
    targets = []
    weights = []
    sample_to_window = []
    
    for window_idx, (start_idx, end_idx) in enumerate(time_windows):
        _validate_window(window_idx, start_idx, end_idx)
        for indices in build_indices(start_idx, end_idx):
            if flatten_sensors:
                sequence = [
                    [
                        float(sensor_data_dict[f"sensor_{sid}"].iloc[t]['force']),
                        float(sensor_data_dict[f"sensor_{sid}"].iloc[t]['displacement']),
                    ]
                    for t in indices
                    for sid in sensor_ids
                ]
                sample_arr = np.array(sequence, dtype=np.float32).reshape(len(indices), len(sensor_ids) * 2)
            else:
                sequence = []
                for sid in sensor_ids:
                    sensor_key = f"sensor_{sid}"
                    sensor_df = sensor_data_dict[sensor_key]
                    sensor_sequence = [
                        [
                            float(sensor_df.iloc[t]['force']),
                            float(sensor_df.iloc[t]['displacement'])
                        ]
                        for t in indices
                    ]
                    sequence.append(sensor_sequence)
                sample_arr = np.array(sequence, dtype=np.float32)
            sample_arr = apply_transforms(sample_arr)
            samples.append(sample_arr)
            sample_to_window.append(window_idx)
            if labels is not None:
                targets.append(float(labels[window_idx]))
                if sample_weights is not None:
                    weights.append(float(sample_weights[window_idx]))
    
    targets_arr = None if labels is None else np.array(targets, dtype=np.float32)
    weights_arr = None if len(weights) == 0 else np.array(weights, dtype=np.float32)
    return TimeSeriesDataset(
        samples,
        targets_arr,
        device=device,
        sample_to_window_idx=sample_to_window,
        sample_weights=weights_arr,
    )

