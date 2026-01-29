"""
Temporal processing modules for sequence modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalLSTM(nn.Module):
    """
    LSTM module for temporal sequence processing.
    
    Processes temporal sequences of node features to capture
    temporal dependencies in the sensor data.
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        """
        Initialize temporal LSTM module.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden dimension size.
            num_layers: Number of LSTM layers.
            dropout: Dropout rate.
        """
        super(TemporalLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input sequence [batch_size, seq_len, input_dim] or
               [num_nodes, seq_len, input_dim].
        
        Returns:
            Output features [batch_size, seq_len, hidden_dim] or
            [num_nodes, seq_len, hidden_dim].
        """
        lstm_out, _ = self.lstm(x)
        return lstm_out


class TemporalAggregator(nn.Module):
    """
    Aggregates temporal outputs to produce a single representation.
    
    Supports different aggregation strategies: last, mean, max.
    """
    
    def __init__(self, aggregation='last'):
        """
        Initialize temporal aggregator.
        
        Args:
            aggregation: Aggregation strategy ('last', 'mean', 'max').
        """
        super(TemporalAggregator, self).__init__()
        
        self.aggregation = aggregation
    
    def forward(self, x):
        """
        Aggregate temporal sequence.
        
        Args:
            x: Temporal sequence [..., seq_len, hidden_dim].
        
        Returns:
            Aggregated features [..., hidden_dim].
        """
        if self.aggregation == 'last':
            return x[:, -1, :]  # Take last time step
        elif self.aggregation == 'mean':
            return x.mean(dim=1)  # Mean over time
        elif self.aggregation == 'max':
            return x.max(dim=1)[0]  # Max over time
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")


class NodeAggregator(nn.Module):
    """
    Aggregates node features to produce graph-level representation.
    
    Supports different aggregation strategies: mean, max, sum.
    """
    
    def __init__(self, aggregation='mean'):
        """
        Initialize node aggregator.
        
        Args:
            aggregation: Aggregation strategy ('mean', 'max', 'sum').
        """
        super(NodeAggregator, self).__init__()
        
        self.aggregation = aggregation
    
    def forward(self, x, batch=None):
        """
        Aggregate node features.
        
        Args:
            x: Node features [num_nodes, hidden_dim].
            batch: Batch assignment vector (optional).
        
        Returns:
            Aggregated features [batch_size, hidden_dim] or [1, hidden_dim].
        """
        if batch is not None:
            from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
            
            if self.aggregation == 'mean':
                return global_mean_pool(x, batch)
            elif self.aggregation == 'max':
                return global_max_pool(x, batch)
            elif self.aggregation == 'sum':
                return global_add_pool(x, batch)
            else:
                raise ValueError(f"Unknown aggregation: {self.aggregation}")
        else:
            if self.aggregation == 'mean':
                return x.mean(dim=0, keepdim=True)
            elif self.aggregation == 'max':
                return x.max(dim=0, keepdim=True)[0]
            elif self.aggregation == 'sum':
                return x.sum(dim=0, keepdim=True)
            else:
                raise ValueError(f"Unknown aggregation: {self.aggregation}")

