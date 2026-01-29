"""
Graph construction modules based on physical principles.

This module provides different graph construction strategies based on:
1. Spatial proximity (distance-based)
2. Stress propagation range (physics-based)
3. Loading direction (directional graphs for rolling load)
"""

from typing import Literal, Optional, Tuple

import numpy as np
import torch


class GraphBuilder:
    """
    Builds graph structures for sensor networks based on physical principles.
    """
    
    def __init__(self, sensor_positions: np.ndarray, 
                 connection_type: Literal['adjacent', 'distance', 'stress_propagation', 'directional'] = 'distance',
                 max_distance: Optional[float] = None,
                 stress_decay_factor: float = 0.5):
        """
        Initialize graph builder.
        
        Args:
            sensor_positions: Array of sensor positions in mm [num_sensors].
            connection_type: Type of graph connection strategy.
            max_distance: Maximum distance for edge connection (mm). 
                         If None, uses 2x average sensor spacing.
            stress_decay_factor: Decay factor for stress propagation (0-1).
        """
        self.sensor_positions = sensor_positions
        self.connection_type = connection_type
        self.num_sensors = len(sensor_positions)
        
        # Calculate average sensor spacing
        if len(sensor_positions) > 1:
            avg_spacing = np.mean(np.diff(np.sort(sensor_positions)))
        else:
            avg_spacing = 35.0  # Default spacing from ref.m
        
        # Set default max_distance if not provided
        if max_distance is None:
            # Stress typically propagates 2-3x the contact area
            # Contact area is 35mm x 60mm, so stress range ~70-105mm
            # Use 2x average spacing as default
            self.max_distance = 2 * avg_spacing
        else:
            self.max_distance = max_distance
        
        self.stress_decay_factor = stress_decay_factor
    
    def compute_distance_matrix(self) -> np.ndarray:
        """
        Computes pairwise distance matrix between sensors.
        
        Returns:
            Distance matrix [num_sensors, num_sensors] in mm.
        """
        positions = self.sensor_positions.reshape(-1, 1)
        distances = np.abs(positions - positions.T)
        return distances
    
    def build_adjacent_graph(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Builds graph with edges only between adjacent sensors (linear chain).
        
        This is the simplest topology, suitable for 1D sensor arrays.
        
        Returns:
            Tuple of (edge_index, edge_attr).
            edge_index: [2, num_edges]
            edge_attr: Optional edge weights [num_edges]
        """
        edges = []
        
        # Create bidirectional edges between adjacent sensors
        for i in range(self.num_sensors - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        return edge_index, None
    
    def build_distance_based_graph(self, threshold: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Builds graph based on spatial distance between sensors.
        
        Creates edges between sensors within a distance threshold.
        Edge weights are inversely proportional to distance (closer = stronger connection).
        
        Args:
            threshold: Distance threshold for edge creation. If None, uses max_distance.
        
        Returns:
            Tuple of (edge_index, edge_attr).
            edge_index: [2, num_edges]
            edge_attr: Edge weights [num_edges] (normalized inverse distance)
        """
        if threshold is None:
            threshold = self.max_distance
        
        distance_matrix = self.compute_distance_matrix()
        
        edges = []
        edge_weights = []
        
        for i in range(self.num_sensors):
            for j in range(self.num_sensors):
                if i != j and distance_matrix[i, j] <= threshold:
                    edges.append([i, j])
                    # Weight inversely proportional to distance
                    # Add small epsilon to avoid division by zero
                    weight = 1.0 / (distance_matrix[i, j] + 1e-6)
                    edge_weights.append(weight)
        
        if len(edges) == 0:
            # Fallback to adjacent graph if no edges found
            return self.build_adjacent_graph()
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float32)
        
        # Normalize edge weights
        edge_attr = edge_attr / edge_attr.max()
        
        return edge_index, edge_attr
    
    def build_stress_propagation_graph(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Builds graph based on stress propagation physics.
        
        In asphalt materials, stress propagates from loading points with exponential decay.
        This creates edges with weights based on stress propagation model:
        σ(r) = σ₀ * exp(-r/λ)
        where r is distance and λ is decay length.
        
        Returns:
            Tuple of (edge_index, edge_attr).
            edge_index: [2, num_edges]
            edge_attr: Edge weights [num_edges] (stress propagation strength)
        """
        distance_matrix = self.compute_distance_matrix()
        
        # Decay length: typical stress decay in asphalt is ~50-100mm
        # Use average sensor spacing as decay length
        avg_spacing = np.mean(np.diff(np.sort(self.sensor_positions)))
        decay_length = avg_spacing / self.stress_decay_factor
        
        edges = []
        edge_weights = []
        
        for i in range(self.num_sensors):
            for j in range(self.num_sensors):
                if i != j:
                    distance = distance_matrix[i, j]
                    
                    # Only connect if within reasonable stress propagation range
                    if distance <= self.max_distance:
                        edges.append([i, j])
                        # Stress propagation: exponential decay
                        weight = np.exp(-distance / decay_length)
                        edge_weights.append(weight)
        
        if len(edges) == 0:
            return self.build_adjacent_graph()
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float32)
        
        return edge_index, edge_attr
    
    def build_directional_graph(self, loading_direction: Literal['left_to_right', 'right_to_left'] = 'right_to_left') -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Builds directional graph based on rolling load direction.
        
        Since loading simulates tire rolling, damage propagates in the loading direction.
        This creates directed edges following the loading path.
        
        Args:
            loading_direction: Direction of rolling load.
                              'right_to_left': sensors 1->11 (as per user description)
                              'left_to_right': sensors 11->1
        
        Returns:
            Tuple of (edge_index, edge_attr).
            edge_index: [2, num_edges] (directed)
            edge_attr: Optional edge weights [num_edges]
        """
        edges = []
        edge_weights = []
        
        # Sort sensor indices by position
        sorted_indices = np.argsort(self.sensor_positions)
        
        if loading_direction == 'right_to_left':
            # Forward direction: right to left (sensor 1 -> 11)
            for idx in range(len(sorted_indices) - 1):
                i = sorted_indices[idx]
                j = sorted_indices[idx + 1]
                edges.append([i, j])
                # Weight based on distance (closer = stronger)
                distance = abs(self.sensor_positions[j] - self.sensor_positions[i])
                edge_weights.append(1.0 / (distance + 1e-6))
            
            # Also allow backward propagation (damage can affect previous positions)
            # but with lower weight
            for idx in range(len(sorted_indices) - 1):
                i = sorted_indices[idx + 1]
                j = sorted_indices[idx]
                edges.append([i, j])
                distance = abs(self.sensor_positions[j] - self.sensor_positions[i])
                edge_weights.append(0.3 / (distance + 1e-6))  # Weaker backward connection
        else:
            # Left to right
            for idx in range(len(sorted_indices) - 1):
                i = sorted_indices[idx + 1]
                j = sorted_indices[idx]
                edges.append([i, j])
                distance = abs(self.sensor_positions[j] - self.sensor_positions[i])
                edge_weights.append(1.0 / (distance + 1e-6))
            
            for idx in range(len(sorted_indices) - 1):
                i = sorted_indices[idx]
                j = sorted_indices[idx + 1]
                edges.append([i, j])
                distance = abs(self.sensor_positions[j] - self.sensor_positions[i])
                edge_weights.append(0.3 / (distance + 1e-6))
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float32)
        edge_attr = edge_attr / edge_attr.max()  # Normalize
        
        return edge_index, edge_attr
    
    def build(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Builds graph based on specified connection type.
        
        Returns:
            Tuple of (edge_index, edge_attr).
        """
        if self.connection_type == 'adjacent':
            return self.build_adjacent_graph()
        elif self.connection_type == 'distance':
            return self.build_distance_based_graph()
        elif self.connection_type == 'stress_propagation':
            return self.build_stress_propagation_graph()
        elif self.connection_type == 'directional':
            return self.build_directional_graph()
        else:
            raise ValueError(f"Unknown connection type: {self.connection_type}")


if __name__ == '__main__':
    # Test graph builder
    from Libs.config import load_dataset_config
    
    config = load_dataset_config()
    sensors_config = config.get('sensors', {})
    sensor_positions_dict = sensors_config.get('sensor_positions', {})
    positions = []
    for i in range(1, 12):
        pos = sensor_positions_dict.get(i, sensor_positions_dict.get(str(i), 0))
        pos_float = float(pos)
        if pos_float > 0:
            positions.append(pos_float)
    sensor_positions = np.array(positions)
    
    print("Testing Graph Builder")
    print("=" * 60)
    
    # Test different connection types
    for conn_type in ['adjacent', 'distance', 'stress_propagation', 'directional']:
        builder = GraphBuilder(sensor_positions, connection_type=conn_type)
        edge_index, edge_attr = builder.build()
        
        print(f"\n{conn_type.upper()} Graph:")
        print(f"  Edges: {edge_index.shape[1]}")
        print(f"  Edge weights: {'Yes' if edge_attr is not None else 'No'}")
        if edge_attr is not None:
            print(f"  Weight range: [{edge_attr.min():.4f}, {edge_attr.max():.4f}]")

