"""
Positional Encoding
Based on "Attention is All You Need" (Vaswani et al., 2017)
"""

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Compute sinusoidal positional encoding.
    
    Provides positional information to the model since Transformer
    has no inherent notion of sequence order.
    """

    def __init__(self, d_model: int, max_len: int = 5000, device: torch.device = None):
        """
        Initialize Positional Encoding.
        
        Args:
            d_model: Dimension of model (must be even)
            max_len: Maximum sequence length
            device: Device to create encoding on
        """
        super(PositionalEncoding, self).__init__()
        
        # Ensure d_model is even for sin/cos pairs
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")

        # Same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # We don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # Compute positional encoding to consider positional information of words
        
        # Register as buffer so it moves with model
        self.register_buffer('pe', self.encoding)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Get positional encoding for input sequence.
        
        Args:
            x: Input tensor (used to get sequence length)
                Shape: (batch_size, seq_len) or (batch_size, seq_len, d_model)
            offset: Position offset for incremental decoding (default: 0)
        
        Returns:
            Positional encoding
                Shape: (seq_len, d_model)
        """
        if x.dim() == 2:
            batch_size, seq_len = x.size()
        else:
            batch_size, seq_len = x.size(0), x.size(1)
        
        return self.pe[offset:offset+seq_len, :]
        # Returns: (seq_len, d_model)
        # Will be added with tok_emb: (batch_size, seq_len, d_model)