"""Transformer model for time series fatigue life prediction."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 50000):
        """Initialize positional encoding."""
        super(PositionalEncoding, self).__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        seq_len = x.size(0)
        if seq_len <= self.max_len:
            return x + self.pe[:seq_len, :]
        else:
            position = torch.arange(0, seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.d_model, 2, device=x.device).float() * (-np.log(10000.0) / self.d_model))
            pe_dynamic = torch.zeros(seq_len, self.d_model, device=x.device)
            pe_dynamic[:, 0::2] = torch.sin(position * div_term)
            pe_dynamic[:, 1::2] = torch.cos(position * div_term)
            pe_dynamic = pe_dynamic.unsqueeze(0).transpose(0, 1)
            return x + pe_dynamic


class TransformerModel(nn.Module):
    """
    Transformer model for time series regression.
    
    Uses TransformerEncoder with positional encoding.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dropout_rate: float = 0.2,
        output_dim: int = 1,
    ):
        """Initialize Transformer model."""
        super(TransformerModel, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout_rate,
            batch_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(d_model // 2, output_dim)
        
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            lengths: Optional tensor of sequence lengths for masking
            
        Returns:
            Predictions of shape [batch_size, output_dim]
        """
        x = x.contiguous()
        x = self.input_projection(x)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        
        if lengths is not None:
            lengths = lengths.contiguous()
            seq_len = x.size(0)
            batch_size = x.size(1)
            positions = torch.arange(seq_len, device=x.device).unsqueeze(1)
            mask = positions < lengths.unsqueeze(0)
            mask = mask.float().unsqueeze(-1)
            x = (x * mask).sum(dim=0) / lengths.float().unsqueeze(-1)
        else:
            x = x.mean(dim=0)
        
        out = F.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class TransformerWrapper:
    """Wrapper class for Transformer model with training and evaluation utilities."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dropout_rate: float = 0.2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        criterion: Optional[nn.Module] = None,
    ):
        """Initialize Transformer wrapper."""
        self.device = device
        self.model = TransformerModel(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
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
        """Train the Transformer model."""
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
            'input_dim': self.model.input_dim,
            'd_model': self.model.d_model,
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load model from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = True
        print(f"Model loaded from {filepath}")

