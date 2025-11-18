"""
Multi-Head Attention
Based on "Attention is All You Need" (Vaswani et al., 2017)
"""

import torch
import torch.nn as nn
from typing import Optional

from .scale_dot_product_attention import ScaleDotProductAttention


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    
    Allows the model to jointly attend to information from different representation
    subspaces at different positions.
    """

    def __init__(self, d_model: int, n_head: int, drop_prob: float = 0.1):
        """
        Initialize Multi-Head Attention.
        
        Args:
            d_model: Model dimension (must be divisible by n_head)
            n_head: Number of attention heads
            drop_prob: Dropout probability
        """
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        
        self.d_model = d_model
        self.n_head = n_head
        self.d_tensor = d_model // n_head
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.w_concat = nn.Linear(d_model, d_model)
        
        # Attention and dropout
        self.attention = ScaleDotProductAttention()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through multi-head attention.
        
        Args:
            q: Query tensor
                Shape: (batch_size, length, d_model)
            k: Key tensor
                Shape: (batch_size, length, d_model)
            v: Value tensor
                Shape: (batch_size, length, d_model)
            mask: Attention mask
                Shape: (batch_size, length, length) or (batch_size, 1, length, length)
                Will be expanded to (batch_size, n_head, length, length)
        
        Returns:
            output: Multi-head attention output
                Shape: (batch_size, length, d_model)
        """
        # 1. Dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. Split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. Handle mask expansion for multi-head
        if mask is not None:
            batch_size, length_q, length_k = q.size(0), q.size(2), k.size(2)
            
            # Handle different mask shapes
            if mask.dim() == 2:
                # (batch_size, length) -> (batch_size, 1, 1, length) for padding mask
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                # (batch_size, length, length) -> (batch_size, 1, length, length)
                mask = mask.unsqueeze(1)
            elif mask.dim() == 4:
                # Already 4D, check if needs expansion
                pass
            
            # Ensure mask has correct shape for attention: (batch_size, n_head, length_q, length_k)
            # Handle padding mask: (B, 1, 1, L) -> expand to (B, n_head, L, L)
            if mask.dim() == 4:
                if mask.size(1) == 1 and mask.size(2) == 1:
                    # Padding mask: (B, 1, 1, L) -> (B, n_head, L, L)
                    # Expand both query and key dimensions
                    mask = mask.expand(batch_size, self.n_head, length_q, length_k)
                elif mask.size(1) == 1:
                    # (B, 1, L, L) -> (B, n_head, L, L)
                    mask = mask.expand(-1, self.n_head, -1, -1)
                elif mask.size(1) != self.n_head:
                    # Expand head dimension if needed
                    mask = mask.expand(-1, self.n_head, -1, -1)

        # 4. Do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask, dropout=self.dropout)

        # 5. Concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        return out

    def split(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Split tensor by number of head.
        
        Args:
            tensor: Input tensor
                Shape: (batch_size, length, d_model)
        
        Returns:
            Split tensor
                Shape: (batch_size, n_head, length, d_tensor)
        """
        batch_size, length, d_model = tensor.size()
        tensor = tensor.view(batch_size, length, self.n_head, self.d_tensor).transpose(1, 2)
        # Similar to group convolution (split by number of heads)
        return tensor

    def concat(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Inverse function of self.split(tensor).
        
        Args:
            tensor: Input tensor
                Shape: (batch_size, n_head, length, d_tensor)
        
        Returns:
            Concatenated tensor
                Shape: (batch_size, length, d_model)
        """
        batch_size, n_head, length, d_tensor = tensor.size()
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, self.d_model)
        return tensor