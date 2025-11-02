"""
Multi-Head attention for Transformer
"""

import torch
import torch.nn as nn
from .scaled_dot_product import ScaledDotProductAttention
from typing import Optional


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention Mechanism.
    
    Allows the model to jointly attend to information from different representation
    subspaces at different positions.
    
    Architecture:
        - Multiple parallel attention heads
        - Each head computes scaled dot-product attention independently
        - Concatenate all heads and project to output dimension
    
    References:
        Vaswani et al. (2017) "Attention is All You Need" NIPS 2017
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize Multi-Head Attention.
        
        Args:
            embed_dim (int): Model dimension (must be divisible by num_heads)
            num_heads (int): Number of attention heads
            dropout (float): Dropout rate
        """
        super(MultiHeadAttention, self).__init__()
        
        assert embed_dim % num_heads == 0, \
            "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.W_o = nn.Linear(embed_dim, embed_dim)
        
        # Attention and dropout
        self.attention = ScaledDotProductAttention()
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through multi-head attention.
        
        Args:
            query (Tensor): Query tensor
                Shape: (batch_size, seq_len_q, embed_dim)
            key (Tensor): Key tensor
                Shape: (batch_size, seq_len_k, embed_dim)
            value (Tensor): Value tensor
                Shape: (batch_size, seq_len_v, embed_dim)
            mask (Tensor, optional): Attention mask
                Shape: (batch_size, seq_len_q, seq_len_k)
        
        Returns:
            output (Tensor): Multi-head attention output
                Shape: (batch_size, seq_len_q, embed_dim)
        """
        batch_size = query.size(0)
        
        # Linear projections and split into heads
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Reshape for multi-head: (batch_size, seq_len, num_heads, head_dim)
        # Transpose: (batch_size, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Expand mask for multi-head if provided
        if mask is not None:
            # (batch_size, seq_len_q, seq_len_k) -> (batch_size, 1, seq_len_q, seq_len_k)
            mask = mask.unsqueeze(1).expand(batch_size, self.num_heads, -1, -1)
        
        # Apply scaled dot-product attention
        # (batch_size, num_heads, seq_len_q, head_dim)
        attended, _ = self.attention(Q, K, V, mask, self.dropout)
        
        # Concatenate heads
        # (batch_size, num_heads, seq_len_q, head_dim) -> (batch_size, seq_len_q, embed_dim)
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, -1, self.embed_dim
        )
        
        # Output projection
        output = self.W_o(attended)
        
        return output