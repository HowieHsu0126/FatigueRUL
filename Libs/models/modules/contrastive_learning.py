"""
Contrastive learning module for time series data (SimCLR style).
Helps alleviate the problem of insufficient training samples through self-supervised pre-training.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from Libs.data.timeseries_dataset import TimeSeriesDataset


class TimeSeriesAugmentation:
    """Time series data augmentation for contrastive learning."""
    
    def __init__(
        self,
        noise_std: float = 0.01,
        time_warp_factor: float = 0.1,
        mask_ratio: float = 0.15,
        enable_noise: bool = True,
        enable_time_warp: bool = True,
        enable_masking: bool = True,
    ):
        """
        Initialize time series augmentation.
        
        Args:
            noise_std: Standard deviation of Gaussian noise.
            time_warp_factor: Factor for time warping (0.0-1.0).
            mask_ratio: Ratio of timesteps to mask (0.0-1.0).
            enable_noise: Whether to enable noise augmentation.
            enable_time_warp: Whether to enable time warping.
            enable_masking: Whether to enable masking.
        """
        self.noise_std = noise_std
        self.time_warp_factor = time_warp_factor
        self.mask_ratio = mask_ratio
        self.enable_noise = enable_noise
        self.enable_time_warp = enable_time_warp
        self.enable_masking = enable_masking
    
    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to time series."""
        if not self.enable_noise or self.noise_std <= 0:
            return x
        noise = torch.randn_like(x) * self.noise_std
        return x + noise
    
    def time_warp(self, x: torch.Tensor) -> torch.Tensor:
        """Apply time warping to time series."""
        if not self.enable_time_warp or self.time_warp_factor <= 0:
            return x
        
        seq_len, feat_dim = x.shape
        warp_steps = int(seq_len * self.time_warp_factor)
        
        if warp_steps < 2:
            return x
        
        warp_indices = torch.linspace(0, seq_len - 1, seq_len + warp_steps)
        warp_indices = torch.clamp(warp_indices, 0, seq_len - 1).long()
        
        warped = x[warp_indices]
        
        if len(warped) > seq_len:
            indices = torch.linspace(0, len(warped) - 1, seq_len).long()
            warped = warped[indices]
        elif len(warped) < seq_len:
            padding = seq_len - len(warped)
            warped = F.pad(warped, (0, 0, 0, padding), mode='replicate')
        
        return warped
    
    def mask_timesteps(self, x: torch.Tensor) -> torch.Tensor:
        """Mask random timesteps in time series."""
        if not self.enable_masking or self.mask_ratio <= 0:
            return x
        
        seq_len, feat_dim = x.shape
        num_mask = int(seq_len * self.mask_ratio)
        
        if num_mask == 0:
            return x
        
        mask_indices = torch.randperm(seq_len)[:num_mask]
        masked = x.clone()
        masked[mask_indices] = 0.0
        
        return masked
    
    def augment(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random augmentation to time series."""
        augmented = x.clone()
        
        if self.enable_noise and torch.rand(1).item() > 0.5:
            augmented = self.add_noise(augmented)
        
        if self.enable_time_warp and torch.rand(1).item() > 0.5:
            augmented = self.time_warp(augmented)
        
        if self.enable_masking and torch.rand(1).item() > 0.5:
            augmented = self.mask_timesteps(augmented)
        
        return augmented


class ContrastiveEncoder(nn.Module):
    """Encoder network for contrastive learning."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        projection_dim: int = 64,
        encoder_type: str = 'lstm',
        bidirectional: bool = True,
    ):
        """
        Initialize contrastive encoder.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden dimension size.
            num_layers: Number of RNN layers.
            dropout: Dropout rate.
            projection_dim: Projection head dimension.
            encoder_type: Type of encoder ('lstm' or 'gru').
            bidirectional: Whether to use bidirectional RNN.
        """
        super(ContrastiveEncoder, self).__init__()
        
        self.encoder_type = encoder_type
        self.bidirectional = bidirectional
        
        if encoder_type == 'lstm':
            self.encoder = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
            )
        elif encoder_type == 'gru':
            self.encoder = nn.GRU(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
        
        output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.projection_head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
        )
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim].
            lengths: Optional sequence lengths.
        
        Returns:
            Projected features [batch_size, projection_dim].
        """
        rnn_out, h_n = self.encoder(x)
        
        if self.bidirectional:
            forward_hidden = h_n[-2]
            backward_hidden = h_n[-1]
            hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        else:
            hidden = h_n[-1]
        
        if hidden.dim() == 3:
            hidden = hidden.squeeze(0)
        
        projected = self.projection_head(hidden)
        projected = F.normalize(projected, p=2, dim=1)
        
        return projected


class ContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss (SimCLR style)."""
    
    def __init__(self, temperature: float = 0.07):
        """
        Initialize contrastive loss.
        
        Args:
            temperature: Temperature parameter for softmax.
        """
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            z1: Projected features from first augmentation [batch_size, projection_dim].
            z2: Projected features from second augmentation [batch_size, projection_dim].
        
        Returns:
            Contrastive loss value.
        """
        batch_size = z1.size(0)
        
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        
        labels = torch.arange(batch_size, device=z1.device)
        
        similarity_matrix = torch.matmul(z1, z2.T) / self.temperature
        
        loss_12 = F.cross_entropy(similarity_matrix, labels)
        loss_21 = F.cross_entropy(similarity_matrix.T, labels)
        
        loss = (loss_12 + loss_21) / 2.0
        
        return loss


class SimCLRTrainer:
    """SimCLR-style contrastive learning trainer."""
    
    def __init__(
        self,
        encoder: ContrastiveEncoder,
        augmentation: TimeSeriesAugmentation,
        device: str = 'cuda',
    ):
        """
        Initialize SimCLR trainer.
        
        Args:
            encoder: Contrastive encoder network.
            augmentation: Time series augmentation module.
            device: Device to run training on.
        """
        self.encoder = encoder.to(device)
        self.augmentation = augmentation
        self.device = device
        self.criterion = ContrastiveLoss()
    
    def pretrain(
        self,
        train_data: TimeSeriesDataset,
        epochs: int = 50,
        lr: float = 0.001,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Pre-train encoder using contrastive learning.
        
        Args:
            train_data: Training dataset.
            epochs: Number of pre-training epochs.
            lr: Learning rate.
            batch_size: Batch size.
            verbose: Whether to print progress.
        
        Returns:
            Dictionary with loss history.
        """
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        
        def collate_fn(batch):
            features_list = [item[0] for item in batch]
            lengths = torch.tensor([feat.shape[0] for feat in features_list], dtype=torch.long)
            max_len = lengths.max().item()
            
            batch_size = len(features_list)
            input_dim = features_list[0].shape[1]
            dtype = features_list[0].dtype
            
            padded_features = torch.zeros(batch_size, max_len, input_dim, dtype=dtype, device=self.device)
            
            for i, feat in enumerate(features_list):
                seq_len = feat.shape[0]
                feat_tensor = torch.as_tensor(feat, dtype=dtype, device=self.device)
                padded_features[i, :seq_len, :] = feat_tensor
            
            lengths = lengths.to(self.device)
            return padded_features, lengths
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        
        loss_history = []
        
        for epoch in range(epochs):
            total_loss = 0
            self.encoder.train()
            
            for features, lengths in train_loader:
                optimizer.zero_grad()
                
                aug1_list = []
                aug2_list = []
                for i in range(features.size(0)):
                    feat = features[i]
                    aug1_list.append(self.augmentation.augment(feat))
                    aug2_list.append(self.augmentation.augment(feat))
                
                aug1 = torch.stack(aug1_list)
                aug2 = torch.stack(aug2_list)
                
                z1 = self.encoder(aug1, lengths)
                z2 = self.encoder(aug2, lengths)
                
                loss = self.criterion(z1, z2)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            loss_history.append(avg_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Contrastive Pre-training Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return {'pretrain_loss': loss_history}
    
    def get_encoder_state_dict(self) -> Dict[str, torch.Tensor]:
        """Returns the state dict of the trained encoder (without projection head)."""
        return self.encoder.encoder.state_dict()
