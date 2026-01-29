"""LSTM model for time series fatigue life prediction."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

from Libs.data.timeseries_dataset import TimeSeriesDataset


def collate_variable_length(
    batch: List[Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]],
    target_device: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Custom collate function for variable-length sequences.
    
    Pads sequences to the same length and returns lengths for pack_padded_sequence.
    
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
    def collate_fn(batch: List[Tuple[torch.Tensor, Optional[torch.Tensor]]]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return collate_variable_length(batch, target_device=device)
    return collate_fn


class LSTMModel(nn.Module):
    """
    LSTM model for time series regression.
    
    Processes multi-sensor time series data to predict RUL (Remaining Useful Life).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout_rate: float = 0.3,
        bidirectional: bool = True,
        output_dim: int = 1,
    ):
        """
        Initialize LSTM model.
        
        Args:
            input_dim: Input feature dimension (num_sensors * 2)
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            dropout_rate: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
            output_dim: Output dimension (1 for RUL prediction)
        """
        super(LSTMModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.fc1 = nn.Linear(lstm_output_dim, lstm_output_dim // 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(lstm_output_dim // 2, output_dim)
        
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            lengths: Optional tensor of sequence lengths for pack_padded_sequence
            
        Returns:
            Predictions of shape [batch_size, output_dim]
        """
        self.lstm.flatten_parameters()
        
        if lengths is not None:
            x = x.contiguous()
            device = x.device
            lengths_cpu = lengths.cpu().long()
            packed_input = pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
            packed_data = packed_input.data.to(device).contiguous()
            packed_input = torch.nn.utils.rnn.PackedSequence(
                packed_data,
                packed_input.batch_sizes,
                packed_input.sorted_indices,
                packed_input.unsorted_indices
            )
            del packed_data
            cudnn_enabled = torch.backends.cudnn.enabled
            torch.backends.cudnn.enabled = False
            packed_output, (h_n, c_n) = self.lstm(packed_input)
            torch.backends.cudnn.enabled = cudnn_enabled
            del packed_input, packed_output
            torch.cuda.empty_cache()
        else:
            x = x.contiguous()
            lstm_out, (h_n, c_n) = self.lstm(x)
            del lstm_out
        
        if self.bidirectional:
            forward_hidden = h_n[-2]
            backward_hidden = h_n[-1]
            hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
            del forward_hidden, backward_hidden
        else:
            hidden = h_n[-1]
        
        del h_n, c_n
        out = F.relu(self.fc1(hidden))
        del hidden
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class LSTMWrapper:
    """
    Wrapper class for LSTM model with training and evaluation utilities.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout_rate: float = 0.3,
        bidirectional: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        criterion: Optional[nn.Module] = None,
    ):
        """
        Initialize LSTM wrapper.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            dropout_rate: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
            device: Device to run model on
        """
        self.device = device
        self.model = LSTMModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            bidirectional=bidirectional,
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
    ) -> Dict[str, List[float]]:
        """
        Train the LSTM model.
        
        Args:
            train_data: Training dataset
            val_data: Optional validation dataset
            epochs: Number of training epochs
            lr: Learning rate
            batch_size: Batch size
            verbose: Whether to print progress
            early_stopping: Early stopping configuration
            scheduler_config: Learning rate scheduler configuration
            
        Returns:
            Dictionary with training and validation loss history
        """
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
                if weights is not None:
                    loss = (loss_raw.squeeze() * weights).mean()
                else:
                    loss = loss_raw.mean()
                
                loss.backward()
                
                if gradient_clipping is not None and gradient_clipping.get('enable', False):
                    max_norm = gradient_clipping.get('max_norm', 1.0)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                
                self.optimizer.step()
                
                total_train_loss += loss.item()
                
                del predictions, loss
                torch.cuda.empty_cache()
            
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
                        
                        del predictions, loss_raw, loss_val
                        torch.cuda.empty_cache()
                
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
            
            torch.cuda.empty_cache()
        
        self.is_trained = True
        return {'train_loss_history': train_loss_history, 'val_loss_history': val_loss_history}

    def predict(self, data: TimeSeriesDataset, batch_size: int = 32) -> np.ndarray:
        """
        Make predictions on dataset.
        
        Args:
            data: Dataset to predict on
            batch_size: Batch size for prediction
            
        Returns:
            Array of predictions
        """
        self.model.eval()
        predictions = []
        
        collate_fn = make_collate_fn(self.device)
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        with torch.no_grad():
            for features, lengths, _, _ in data_loader:
                pred = self.model(features, lengths)
                predictions.append(pred.cpu().numpy())
                
                del pred, features, lengths
                torch.cuda.empty_cache()
        
        return np.concatenate(predictions, axis=0).flatten()

    def save_model(self, filepath: str):
        """Save model to file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.model.input_dim,
            'hidden_dim': self.model.hidden_dim,
            'num_layers': self.model.num_layers,
            'dropout_rate': self.model.lstm.dropout if self.model.num_layers > 1 else 0,
            'bidirectional': self.model.bidirectional,
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load model from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model = LSTMModel(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            num_layers=checkpoint['num_layers'],
            dropout_rate=checkpoint['dropout_rate'],
            bidirectional=checkpoint['bidirectional'],
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = True
        print(f"Model loaded from {filepath}")

