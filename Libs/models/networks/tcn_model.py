"""Temporal Convolutional Network (TCN) for time series fatigue life prediction."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader

from Libs.data.timeseries_dataset import TimeSeriesDataset


def collate_variable_length(
    batch: List[Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]],
    target_device: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Custom collate function for variable-length sequences.
    
    Pads sequences to the same length.
    
    Args:
        batch: List of (features, target) tuples where features have shape [seq_len, input_dim]
        target_device: Target device to move tensors to. If None, uses the device of first feature.
        
    Returns:
        Tuple of (padded_features, lengths, targets) where:
        - padded_features: [batch_size, max_seq_len, input_dim]
        - lengths: [batch_size] tensor of sequence lengths
        - targets: [batch_size] tensor of target values or None
    """
    features_list = [item[0] for item in batch]
    targets_list = [item[1] for item in batch]
    weights_list = [item[2] for item in batch]
    
    lengths = torch.tensor([feat.shape[0] for feat in features_list], dtype=torch.long)
    max_len = lengths.max().item()
    
    batch_size = len(features_list)
    input_dim = features_list[0].shape[1]
    dtype = features_list[0].dtype
    
    device = target_device if target_device is not None else features_list[0].device
    
    padded_features_cpu = torch.zeros(batch_size, max_len, input_dim, dtype=dtype)
    
    for i, feat in enumerate(features_list):
        seq_len = feat.shape[0]
        feat_cpu = feat.cpu()
        padded_features_cpu[i, :seq_len, :] = feat_cpu
        del feat_cpu
    
    padded_features = padded_features_cpu.to(device).contiguous()
    del padded_features_cpu
    lengths = lengths.to(device).contiguous()
    
    targets = None
    if all(t is not None for t in targets_list):
        targets = torch.stack(targets_list).to(device)
    weights = None
    if any(w is not None for w in weights_list):
        weights_stack = [torch.as_tensor(1.0, dtype=torch.float32) if w is None else w for w in weights_list]
        weights = torch.stack(weights_stack).to(device)
    
    del features_list
    return padded_features, lengths, targets, weights


def make_collate_fn(device: str):
    """
    Create a collate function bound to a specific device.
    
    Args:
        device: Target device for the collate function
        
    Returns:
        A collate function that moves data to the specified device
    """
    def collate_fn(batch: List[Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        return collate_variable_length(batch, target_device=device)
    return collate_fn


class Chomp1d(nn.Module):
    """Remove padding from the end of sequence."""

    def __init__(self, chomp_size: int):
        """Initialize chomp layer."""
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Remove padding."""
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Temporal block with dilated causal convolution."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ):
        """Initialize temporal block."""
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNModel(nn.Module):
    """
    Temporal Convolutional Network for time series regression.
    
    Uses dilated causal convolutions with residual connections.
    """

    def __init__(
        self,
        input_dim: int,
        num_channels: list = None,
        kernel_size: int = 3,
        dropout_rate: float = 0.2,
        output_dim: int = 1,
    ):
        """Initialize TCN model."""
        super(TCNModel, self).__init__()
        
        if num_channels is None:
            num_channels = [64, 128, 256]
        
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout_rate)]

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_dim)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            lengths: Optional tensor of sequence lengths (not used for TCN, kept for compatibility)
            
        Returns:
            Predictions of shape [batch_size, output_dim]
        """
        x = x.contiguous()
        x = x.transpose(1, 2)
        y = self.network(x)
        y = y[:, :, -1]
        return self.fc(y)


class TCNWrapper:
    """Wrapper class for TCN model with training and evaluation utilities."""

    def __init__(
        self,
        input_dim: int,
        num_channels: list = None,
        kernel_size: int = 3,
        dropout_rate: float = 0.2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        criterion: Optional[nn.Module] = None,
    ):
        """Initialize TCN wrapper."""
        self.device = device
        self.model = TCNModel(
            input_dim=input_dim,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
        ).to(device)
        
        self.optimizer = None
        self.criterion = criterion if criterion is not None else nn.MSELoss(reduction='none')
        self.is_trained = False

    def train(
        self,
        train_data: TimeSeriesDataset,
        val_data: Optional[TimeSeriesDataset] = None,
        epochs: int = 100,
        lr: float = 0.001,
        batch_size: int = 32,
        verbose: bool = True,
        early_stopping: Optional[Dict[str, Any]] = None,
        scheduler_config: Optional[Dict[str, Any]] = None,
        gradient_clipping: Optional[Dict[str, Any]] = None,
        warmup_epochs: int = 0,
    ) -> Dict[str, list]:
        """Train the TCN model."""
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        
        collate_fn = make_collate_fn(self.device)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn) if val_data else None
        
        train_loss_history = []
        val_loss_history = []
        
        best_val_loss = float('inf')
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
            
            total_train_loss = 0
            self.model.train()
            
            for features, lengths, targets, weights in train_loader:
                self.optimizer.zero_grad()
                
                predictions = self.model(features, lengths)
                loss_raw = self.criterion(predictions, targets.unsqueeze(1))
                loss = (loss_raw.squeeze() * weights).mean() if weights is not None else loss_raw.mean()
                
                loss.backward()
                
                if gradient_clipping is not None and gradient_clipping.get('enable', False):
                    max_norm = gradient_clipping.get('max_norm', 1.0)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                
                self.optimizer.step()
                
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            train_loss_history.append(avg_train_loss)
            
            val_loss = None
            if val_loader is not None:
                self.model.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for features, lengths, targets, weights in val_loader:
                        predictions = self.model(features, lengths)
                        loss_raw = self.criterion(predictions, targets.unsqueeze(1))
                        loss_val = (loss_raw.squeeze() * weights).mean() if weights is not None else loss_raw.mean()
                        total_val_loss += loss_val.item()
                
                val_loss = total_val_loss / len(val_loader)
                val_loss_history.append(val_loss)
                
                if scheduler:
                    scheduler.step(val_loss)
                
                if enable_early_stopping:
                    if val_loss < best_val_loss - min_delta:
                        best_val_loss = val_loss
                        patience_counter = 0
                        if restore_best:
                            best_model_state = self.model.state_dict().copy()
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            if verbose:
                                print(f"Early stopping at epoch {epoch+1}")
                            if restore_best and 'best_model_state' in locals():
                                self.model.load_state_dict(best_model_state)
                            break
            else:
                if scheduler and val_loss is not None:
                    scheduler.step(val_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                val_str = f", Val Loss: {val_loss:.4f}" if val_loss is not None else ""
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}{val_str}")
        
        self.is_trained = True
        return {'train_loss_history': train_loss_history, 'val_loss_history': val_loss_history}

    def predict(self, data: TimeSeriesDataset, batch_size: int = 32) -> np.ndarray:
        """Make predictions on dataset."""
        self.model.eval()
        predictions = []
        
        collate_fn = make_collate_fn(self.device)
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        with torch.no_grad():
            for features, lengths, _, _ in data_loader:
                pred = self.model(features, lengths)
                predictions.append(pred.cpu().numpy())
        
        return np.concatenate(predictions, axis=0).flatten()

    def save_model(self, filepath: str):
        """Save model to file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load model from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = True
        print(f"Model loaded from {filepath}")

