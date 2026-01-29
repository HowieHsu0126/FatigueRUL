"""Dataset utilities for time-series and graph-structured fatigue data."""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from Libs.data.graph_builder import GraphBuilder


class TimeSeriesDataset(Dataset):
    """Provides PyTorch access to processed time-series samples and labels."""

    def __init__(
        self,
        samples: Sequence[Any],
        targets: Optional[Sequence[Any]] = None,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
    ) -> None:
        self.samples = torch.as_tensor(np.asarray(samples), dtype=torch.float32)
        self.targets = None if targets is None else torch.as_tensor(np.asarray(targets), dtype=torch.float32)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """Returns the number of samples stored in the dataset."""
        return int(self.samples.shape[0])

    def __getitem__(self, index: int):
        """Fetches a single item, optionally applying feature and target transforms."""
        feature = self.samples[index]
        feature = self.transform(feature) if self.transform is not None else feature
        if self.targets is None:
            return feature
        target = self.targets[index]
        target = self.target_transform(target) if self.target_transform is not None else target
        return feature, target


class GraphDataset(Dataset):
    """Builds and stores graph samples derived from sensor measurements."""

    def __init__(
        self,
        sensor_positions: np.ndarray,
        graph_type: str = "stress_propagation",
        device: str = "cpu",
    ) -> None:
        self.sensor_positions = np.asarray(sensor_positions)
        self.graph_type = graph_type
        self.device = device
        self.num_nodes = int(self.sensor_positions.shape[0])

        self.graph_builder = GraphBuilder(self.sensor_positions, connection_type=graph_type)
        self.edge_index, self.edge_attr = self.graph_builder.build()
        self.edge_index = self.edge_index.to(device)
        if self.edge_attr is not None:
            self.edge_attr = self.edge_attr.to(device)

        self.graphs: List[Data] = []

    def __len__(self) -> int:
        """Returns the number of cached graph samples."""
        return len(self.graphs)

    def __getitem__(self, index: int) -> Data:
        """Returns a single cached graph sample."""
        return self.graphs[index]

    def prepare_graph_data(
        self,
        sensor_data_dict: Dict[str, pd.DataFrame],
        time_windows: Iterable[Tuple[int, int]],
        labels: Optional[Sequence[float]] = None,
    ) -> List[Data]:
        """Creates PyG graph objects for each requested time window."""
        sensor_ids = sorted(int(key.split("_")[1]) for key in sensor_data_dict.keys())
        graphs: List[Data] = []

        for idx, (start_idx, end_idx) in enumerate(time_windows):
            node_features: List[List[float]] = []
            for sensor_id in sensor_ids:
                sensor_key = f"sensor_{sensor_id}"
                sensor_df = sensor_data_dict[sensor_key]
                window_slice = sensor_df.iloc[start_idx:end_idx]
                if len(window_slice) > 0:
                    force_data = window_slice["force"].values
                    disp_data = window_slice["displacement"].values
                    
                    force_mean = float(force_data.mean())
                    force_std = float(force_data.std()) if len(force_data) > 1 else 0.0
                    force_max = float(force_data.max())
                    force_min = float(force_data.min())
                    
                    disp_mean = float(disp_data.mean())
                    disp_std = float(disp_data.std()) if len(disp_data) > 1 else 0.0
                    disp_max = float(disp_data.max())
                    disp_min = float(disp_data.min())
                    
                    time_indices = np.arange(len(window_slice))
                    force_slope = float(np.polyfit(time_indices, force_data, 1)[0]) if len(force_data) > 1 else 0.0
                    disp_slope = float(np.polyfit(time_indices, disp_data, 1)[0]) if len(disp_data) > 1 else 0.0
                    
                    feature_values = [
                        force_mean, force_std, force_max, force_min, force_slope,
                        disp_mean, disp_std, disp_max, disp_min, disp_slope
                    ]
                else:
                    feature_values = [0.0] * 10
                node_features.append(feature_values)

            features = torch.tensor(node_features, dtype=torch.float32, device=self.device)
            target_tensor = None
            if labels is not None:
                target_value = labels[idx]
                target_tensor = torch.tensor([[target_value]], dtype=torch.float32, device=self.device)

            if self.edge_attr is not None:
                graph = Data(x=features, edge_index=self.edge_index, edge_attr=self.edge_attr, y=target_tensor)
            else:
                graph = Data(x=features, edge_index=self.edge_index, y=target_tensor)
            graphs.append(graph)

        self.graphs = graphs
        return graphs

    def save_graph_data(
        self,
        graph_data_list: Optional[Sequence[Data]] = None,
        output_dir: str = "Input/graph_processed",
        filename: str = "graph_data.pt",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persists graph samples and metadata for later reuse."""
        target_graphs = list(graph_data_list) if graph_data_list is not None else list(self.graphs)
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)

        serialized_graphs = [graph.to("cpu") for graph in target_graphs]
        package = {
            "graph_data": serialized_graphs,
            "num_samples": len(serialized_graphs),
            "graph_type": self.graph_type,
            "sensor_positions": self.sensor_positions.tolist(),
            "num_nodes": len(self.sensor_positions),
            "edge_index": self.edge_index.cpu(),
        }

        if self.edge_attr is not None:
            package["edge_attr"] = self.edge_attr.cpu()
        if metadata is not None:
            package["metadata"] = metadata

        torch.save(package, filepath)

    @staticmethod
    def load_graph_data(filepath: str, device: str = "cpu") -> Tuple[List[Data], Dict[str, Any]]:
        """Loads graph samples and metadata from disk."""
        loaded = torch.load(filepath, map_location=device)
        graph_data_list = [graph.to(device) for graph in loaded["graph_data"]]

        metadata: Dict[str, Any] = {
            "num_samples": loaded.get("num_samples", len(graph_data_list)),
            "graph_type": loaded.get("graph_type", "unknown"),
            "sensor_positions": loaded.get("sensor_positions", []),
            "num_nodes": loaded.get("num_nodes", 0),
            "edge_index": loaded.get("edge_index"),
            "edge_attr": loaded.get("edge_attr"),
        }
        metadata.update(loaded.get("metadata", {}))

        return graph_data_list, metadata


