"""
Scaled Dot-Product Attention
Based on "Attention is All You Need" (Vaswani et al., 2017)
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class ScaleDotProductAttention(nn.Module):
    """
    Compute scaled dot product attention.
    
    Query: given sentence that we focused on (decoder)
    Key: every sentence to check relationship with Query (encoder)
    Value: every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        dropout: Optional[nn.Module] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through scaled dot-product attention.
        
        Args:
            q: Query tensor
                Shape: (batch_size, n_head, length, d_tensor)
            k: Key tensor
                Shape: (batch_size, n_head, length, d_tensor)
            v: Value tensor
                Shape: (batch_size, n_head, length, d_tensor)
            mask: Attention mask
                Shape: (batch_size, n_head, length, length) or broadcastable
            dropout: Dropout layer (optional)
        
        Returns:
            output: Attention output
                Shape: (batch_size, n_head, length, d_tensor)
            attention: Attention weights
                Shape: (batch_size, n_head, length, length)
        """
        batch_size, n_head, length, d_tensor = k.size()

        # 1. Dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # (batch_size, n_head, d_tensor, length)
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. Apply masking (opt)
        if mask is not None:
            # Mask should be broadcastable: (batch_size, n_head, length, length)
            # Fill masked positions with large negative value
            score = score.masked_fill(mask == 0, float('-inf'))

        # 3. Pass through softmax to make [0, 1] range
        attention = self.softmax(score)
        
        # Apply dropout to attention weights
        if dropout is not None:
            attention = dropout(attention)

        # 4. Multiply with Value
        output = attention @ v

        return output, attention