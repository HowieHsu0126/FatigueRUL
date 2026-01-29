"""
Graph Convolutional Network layers for spatial feature extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNLayer(nn.Module):
    """
    Graph Convolutional Network layer with activation and dropout.
    
    This is a wrapper around PyTorch Geometric's GCNConv that adds
    activation and dropout for better modularity.
    """
    
    def __init__(self, in_channels, out_channels, dropout=0.1, activation='relu'):
        """
        Initialize GCN layer.
        
        Args:
            in_channels: Input feature dimension.
            out_channels: Output feature dimension.
            dropout: Dropout rate.
            activation: Activation function ('relu', 'tanh', or None).
        """
        super(GCNLayer, self).__init__()
        
        self.gcn = GCNConv(in_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        else:
            self.activation = None
    
    def forward(self, x, edge_index):
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, in_channels].
            edge_index: Graph edge indices [2, num_edges].
        
        Returns:
            Output features [num_nodes, out_channels].
        """
        x = self.gcn(x, edge_index)
        
        if self.activation is not None:
            x = self.activation(x)
        
        x = self.dropout(x)
        
        return x


class MultiLayerGCN(nn.Module):
    """
    Multi-layer Graph Convolutional Network.
    
    Stacks multiple GCN layers for deeper feature extraction.
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        """
        Initialize multi-layer GCN.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden dimension size.
            num_layers: Number of GCN layers.
            dropout: Dropout rate.
        """
        super(MultiLayerGCN, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # First layer: input_dim -> hidden_dim
        self.layers.append(GCNLayer(input_dim, hidden_dim, dropout))
        
        # Subsequent layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 1):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim, dropout))
    
    def forward(self, x, edge_index):
        """
        Forward pass through all GCN layers.
        
        Args:
            x: Node features [num_nodes, input_dim].
            edge_index: Graph edge indices [2, num_edges].
        
        Returns:
            Output features [num_nodes, hidden_dim].
        """
        for layer in self.layers:
            x = layer(x, edge_index)
        
        return x

