"""
Prediction head modules for final output generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPPredictionHead(nn.Module):
    """
    Multi-layer perceptron prediction head.
    
    Maps aggregated features to final predictions (e.g., RUL).
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2, dropout=0.1):
        """
        Initialize MLP prediction head.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden dimension size.
            output_dim: Output dimension (default 1 for regression).
            num_layers: Number of MLP layers.
            dropout: Dropout rate.
        """
        super(MLPPredictionHead, self).__init__()
        
        layers = []
        
        # First layer: input_dim -> hidden_dim
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # Output layer: hidden_dim -> output_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input features [batch_size, input_dim] or [1, input_dim].
        
        Returns:
            Predictions [batch_size, output_dim] or [1, output_dim].
        """
        return self.mlp(x)

