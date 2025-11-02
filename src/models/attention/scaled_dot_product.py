"""
Scaled dot-product attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention Mechanism.
    
    Formula:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    References:
        Vaswani et al. (2017) "Attention is All You Need" NIPS 2017
    """
    
    def __init__(self):
        """Initialize Scaled Dot-Product Attention."""
        super(ScaledDotProductAttention, self).__init__()
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        dropout: Optional[nn.Module] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.
        
        Args:
            query (Tensor): Query matrix
                Shape: (batch_size, num_heads, seq_len_q, d_k)
            key (Tensor): Key matrix
                Shape: (batch_size, num_heads, seq_len_k, d_k)
            value (Tensor): Value matrix
                Shape: (batch_size, num_heads, seq_len_v, d_v)
            mask (Tensor, optional): Attention mask
                Shape: (batch_size, 1, seq_len_q, seq_len_k)
            dropout (nn.Module, optional): Dropout layer
        
        Returns:
            output (Tensor): Attention output
                Shape: (batch_size, num_heads, seq_len_q, d_v)
            attention_weights (Tensor): Attention distribution
                Shape: (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        # Compute d_k for scaling
        d_k = query.size(-1)
        
        # Compute attention scores: QK^T / sqrt(d_k)
        # (batch_size, num_heads, seq_len_q, d_k) @ (batch_size, num_heads, d_k, seq_len_k) -> (batch_size, num_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float(-1e9))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout if provided
        if dropout is not None:
            attention_weights = dropout(attention_weights)
        
        # Compute weighted values
        # (batch_size, num_heads, seq_len_q, seq_len_k) @ (batch_size, num_heads, seq_len_v, d_v) -> (batch_size, num_heads, seq_len_q, d_v)
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights