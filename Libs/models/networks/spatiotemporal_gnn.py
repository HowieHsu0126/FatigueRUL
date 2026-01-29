from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from Libs.models.layers.gcn_layer import MultiLayerGCN
from Libs.models.modules.prediction_head import MLPPredictionHead
from Libs.models.modules.temporal_module import (NodeAggregator,
                                                 TemporalAggregator,
                                                 TemporalLSTM)


class TemporalGCN(nn.Module):
    """
    Temporal Graph Convolutional Network for spatio-temporal learning.
    
    This model combines graph convolution for spatial dependencies and
    temporal convolution/LSTM for temporal dependencies.
    """
    
    def __init__(self, num_nodes=9, input_dim=2, hidden_dim=64, 
                 num_layers=2, output_dim=1, dropout=0.1):
        """
        Initialize the TemporalGCN model.
        
        Args:
            num_nodes: Number of sensor nodes in the graph.
            input_dim: Dimension of input features per node (force, displacement).
            hidden_dim: Hidden dimension size.
            num_layers: Number of GCN layers.
            output_dim: Output dimension (RUL prediction).
            dropout: Dropout rate.
        """
        super(TemporalGCN, self).__init__()
        
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Graph convolution layers
        self.gcn = MultiLayerGCN(input_dim, hidden_dim, num_layers, dropout)
        
        # Temporal processing (LSTM)
        self.temporal_lstm = TemporalLSTM(hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.temporal_aggregator = TemporalAggregator(aggregation='last')
        
        # Node aggregation
        self.node_aggregator = NodeAggregator(aggregation='mean')
        
        # Prediction head
        self.prediction_head = MLPPredictionHead(
            hidden_dim, hidden_dim // 2, output_dim, num_layers=2, dropout=dropout
        )
        
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes * batch_size, input_dim] or [num_nodes, seq_len, input_dim]
            edge_index: Graph edge indices [2, num_edges]
            batch: Batch assignment vector (optional)
        
        Returns:
            Predictions [batch_size, output_dim]
        """
        # If x is 3D (nodes, time, features), process temporally first
        if len(x.shape) == 3:
            num_nodes, seq_len, feat_dim = x.shape
            
            # Process each time step through GCN
            temporal_outputs = []
            for t in range(seq_len):
                x_t = x[:, t, :]  # [num_nodes, feat_dim]
                x_t = self.gcn(x_t, edge_index)  # [num_nodes, hidden_dim]
                temporal_outputs.append(x_t)
            
            # Stack temporal outputs: [seq_len, num_nodes, hidden_dim]
            temporal_outputs = torch.stack(temporal_outputs, dim=0)
            
            # Reshape for LSTM: [num_nodes, seq_len, hidden_dim]
            temporal_outputs = temporal_outputs.permute(1, 0, 2)
            
            # Apply LSTM and aggregate temporally
            lstm_out = self.temporal_lstm(temporal_outputs)  # [num_nodes, seq_len, hidden_dim]
            x = self.temporal_aggregator(lstm_out)  # [num_nodes, hidden_dim]
            
            # Aggregate node features
            x = self.node_aggregator(x, batch)  # [batch_size, hidden_dim] or [1, hidden_dim]
        else:
            # 2D input: [num_nodes, features]
            x = self.gcn(x, edge_index)  # [num_nodes, hidden_dim]
            x = self.node_aggregator(x, batch)  # [batch_size, hidden_dim] or [1, hidden_dim]
        
        # Final prediction
        x = self.prediction_head(x)  # [batch_size, output_dim] or [1, output_dim]
        
        return x


class SpatioTemporalGNN:
    """
    Wrapper class for Spatio-Temporal Graph Neural Network model.
    Handles data preparation, training, and evaluation.
    """
    
    def __init__(self, num_nodes=9, input_dim=2, hidden_dim=64, 
                 num_layers=2, device='cuda' if torch.cuda.is_available() else 'cpu',
                 edge_index: Optional[torch.Tensor] = None,
                 edge_attr: Optional[torch.Tensor] = None):
        """
        Initialize the SpatioTemporalGNN wrapper.
        
        Args:
            num_nodes: Number of sensor nodes.
            input_dim: Input feature dimension per node.
            hidden_dim: Hidden dimension size.
            num_layers: Number of GCN layers.
            device: Device to run the model on.
            edge_index: Pre-built graph edge index [2, num_edges]. If None, will be set during training.
            edge_attr: Optional edge attributes [num_edges]. If None, will be set during training.
        """
        self.device = device
        self.num_nodes = num_nodes
        
        self.model = TemporalGCN(
            num_nodes=num_nodes,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        ).to(device)
        
        self._initialize_weights()
        
        self.optimizer = None
        self.criterion = nn.MSELoss()
        
        # Store graph structure if provided
        if edge_index is not None:
            self.edge_index = edge_index.to(device)
            self.edge_attr = edge_attr.to(device) if edge_attr is not None else None
        else:
            self.edge_index = None
            self.edge_attr = None
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier uniform initialization."""
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LSTM) or isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0.0)
                        n = param.size(0)
                        start, end = n // 4, n // 2
                        param.data[start:end].fill_(1.0)
    
    def prepare_graph_data(self, sensor_data_dict: Dict[str, pd.DataFrame],
                          time_windows: List[Tuple[int, int]],
                          labels: Optional[np.ndarray] = None,
                          edge_index: Optional[torch.Tensor] = None,
                          edge_attr: Optional[torch.Tensor] = None) -> List[Data]:
        """
        Prepares graph data from sensor measurements.
        
        Note: This method is deprecated. Use GraphDataset.prepare_graph_data() instead.
        
        Args:
            sensor_data_dict: Dictionary of sensor dataframes.
            time_windows: List of (start_idx, end_idx) tuples for each sample.
            labels: Optional array of labels (RUL values).
            edge_index: Graph edge index [2, num_edges]. Uses self.edge_index if None.
            edge_attr: Optional edge attributes [num_edges]. Uses self.edge_attr if None.
        
        Returns:
            List of PyTorch Geometric Data objects.
        """
        if edge_index is None:
            if self.edge_index is None:
                raise ValueError("edge_index must be provided either as parameter or in __init__")
            edge_index = self.edge_index
        else:
            edge_index = edge_index.to(self.device)
        
        if edge_attr is None:
            edge_attr = self.edge_attr
        else:
            edge_attr = edge_attr.to(self.device) if edge_attr is not None else None
        
        sensor_ids = sorted([int(k.split('_')[1]) for k in sensor_data_dict.keys()])
        num_nodes = len(sensor_ids)
        
        graph_data_list = []
        
        for idx, (start_idx, end_idx) in enumerate(time_windows):
            # Extract time window data for all sensors
            node_features = []
            
            for sid in sensor_ids:
                key = f'sensor_{sid}'
                sensor_df = sensor_data_dict[key]
                
                # Extract force and displacement for this time window
                window_data = sensor_df.iloc[start_idx:end_idx]
                
                # Use mean and std as features (or use raw values if window is small)
                if len(window_data) > 0:
                    force_mean = window_data['force'].mean()
                    disp_mean = window_data['displacement'].mean()
                    node_features.append([force_mean, disp_mean])
                else:
                    node_features.append([0.0, 0.0])
            
            # Convert to tensor: [num_nodes, input_dim]
            x = torch.tensor(node_features, dtype=torch.float32).to(self.device)
            
            # Create graph data object
            if labels is not None:
                y = torch.tensor([[labels[idx]]], dtype=torch.float32).to(self.device)
            else:
                y = None
            
            # Create graph data object with optional edge attributes
            if edge_attr is not None:
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            else:
                data = Data(x=x, edge_index=edge_index, y=y)
            graph_data_list.append(data)
        
        return graph_data_list
    
    def train(self, train_data: List[Data], epochs=100, lr=0.001, 
              batch_size=32, verbose=True, val_data: Optional[List[Data]] = None,
              early_stopping: Optional[Dict[str, Any]] = None,
              scheduler_config: Optional[Dict[str, Any]] = None,
              gradient_clipping: Optional[Dict[str, Any]] = None,
              warmup_epochs: int = 0,
              sample_weights: Optional[Sequence[float]] = None):
        """
        Trains the GNN model with optional validation and early stopping.
        
        Args:
            train_data: List of training graph data objects.
            epochs: Number of training epochs.
            lr: Learning rate.
            batch_size: Batch size for training.
            verbose: Whether to print training progress.
            val_data: Optional validation data for monitoring.
            early_stopping: Optional early stopping configuration dict.
            scheduler_config: Optional learning rate scheduler configuration dict.
            sample_weights: Optional per-sample weights for loss (same length as train_data).
        
        Returns:
            Dictionary with training and validation loss history.
        """
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        
        train_loss_history = []
        val_loss_history = []
        
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        enable_early_stopping = early_stopping is not None and early_stopping.get('enable', False)
        patience = early_stopping.get('patience', 15) if enable_early_stopping else epochs
        min_delta = early_stopping.get('min_delta', 0.001) if enable_early_stopping else 0.0
        restore_best = early_stopping.get('restore_best_weights', True) if enable_early_stopping else False
        
        scheduler = None
        if scheduler_config is not None and scheduler_config.get('type') == 'reduce_on_plateau':
            factor = scheduler_config.get('factor', 0.5)
            patience_sched = scheduler_config.get('patience', 10)
            threshold = scheduler_config.get('threshold', 0.0001)
            cooldown = scheduler_config.get('cooldown', 0)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=factor, patience=patience_sched, 
                threshold=threshold, cooldown=cooldown
            )
        
        for epoch in range(epochs):
            if warmup_epochs > 0:
                warmup_lr = lr * (epoch + 1) / warmup_epochs if epoch < warmup_epochs else lr
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            
            total_loss = 0
            
            self.model.train()
            for i in range(0, len(train_data), batch_size):
                batch_data = train_data[i:i+batch_size]
                
                batch_loss = 0
                for j, data in enumerate(batch_data):
                    self.optimizer.zero_grad()
                    
                    data = data.to(self.device)
                    pred = self.model(data.x, data.edge_index)
                    loss = self.criterion(pred, data.y)
                    if sample_weights is not None:
                        loss = loss * float(sample_weights[i + j])
                    
                    loss.backward()
                    
                    if gradient_clipping is not None and gradient_clipping.get('enable', False):
                        max_norm = gradient_clipping.get('max_norm', 1.0)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                    
                    self.optimizer.step()
                    
                    batch_loss += loss.item()
                
                total_loss += batch_loss / len(batch_data)
            
            avg_train_loss = total_loss / (len(train_data) // batch_size + 1)
            train_loss_history.append(avg_train_loss)
            
            val_loss = None
            if val_data is not None:
                self.model.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for i in range(0, len(val_data), batch_size):
                        batch_data = val_data[i:i+batch_size]
                        batch_val_loss = 0
                        for data in batch_data:
                            data = data.to(self.device)
                            pred = self.model(data.x, data.edge_index)
                            loss = self.criterion(pred, data.y)
                            batch_val_loss += loss.item()
                        total_val_loss += batch_val_loss / len(batch_data)
                
                val_loss = total_val_loss / (len(val_data) // batch_size + 1)
                val_loss_history.append(val_loss)
                
                if scheduler is not None:
                    scheduler.step(val_loss)
                
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if restore_best:
                        best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                if enable_early_stopping and patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}")
        
        if restore_best and best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            if verbose:
                print(f"Restored best model weights (val_loss: {best_val_loss:.4f})")
        
        return {
            'train_loss': train_loss_history,
            'val_loss': val_loss_history if val_data is not None else None
        }
    
    def predict(self, test_data: List[Data]) -> np.ndarray:
        """
        Makes predictions on test data.
        
        Args:
            test_data: List of test graph data objects.
        
        Returns:
            Array of predictions.
        """
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for data in test_data:
                data = data.to(self.device)
                pred = self.model(data.x, data.edge_index)
                predictions.append(pred.cpu().numpy())
        
        return np.array(predictions).flatten()
    
    def save_model(self, filepath: str):
        """Saves the trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_nodes': self.num_nodes,
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Loads a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {filepath}")


if __name__ == '__main__':
    from Libs.data.dataloader import load_arrows_data
    from Libs.data.label_generator import FatigueLabelGenerator
    from Libs.data.physics_processor import PhysicsModel
    
    print("Loading ARROWS data...")
    data = load_arrows_data('Input/raw/data.mat')
    
    if data:
        print("Computing elastic modulus...")
        physics = PhysicsModel()
        modulus_result = physics.compute_elastic_modulus(data['sensors'], data['time'])
        
        print("Generating training dataset...")
        label_gen = FatigueLabelGenerator()
        training_df = label_gen.prepare_training_data(
            modulus_result, 
            data['sensors'], 
            data['time']
        )
        
        print("Preparing graph data for GNN...")
        gnn = SpatioTemporalGNN(num_nodes=9, input_dim=2, hidden_dim=32)
        
        # Prepare time windows (use cycle indices)
        cycle_indices = modulus_result.get('cycle_indices', [])
        labels = training_df['rul'].values
        
        # Filter valid samples
        valid_mask = ~np.isnan(labels)
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) > 0:
            time_windows = [cycle_indices[i] for i in valid_indices if i < len(cycle_indices)]
            valid_labels = labels[valid_indices[:len(time_windows)]]
            
            graph_data = gnn.prepare_graph_data(
                data['sensors'],
                time_windows,
                valid_labels
            )
            
            print(f"Prepared {len(graph_data)} graph samples")
            print("GNN model initialized successfully!")

