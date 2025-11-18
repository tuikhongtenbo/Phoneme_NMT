"""
Transformer Decoder Layer
Based on "Attention is All You Need" (Vaswani et al., 2017)
"""

import torch
import torch.nn as nn
from typing import Optional

from ..layers.multi_head_attention import MultiHeadAttention
from ..layers.position_wise_feed_forward import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    """
    Single decoder layer of Transformer.
    
    Consists of:
    1. Masked multi-head self-attention
    2. Add & Norm (residual connection + layer normalization)
    3. Multi-head encoder-decoder attention
    4. Add & Norm (residual connection + layer normalization)
    5. Position-wise feed-forward network
    6. Add & Norm (residual connection + layer normalization)
    """

    def __init__(self, d_model: int, ffn_hidden: int, n_head: int, drop_prob: float = 0.1):
        """
        Initialize Decoder Layer.
        
        Args:
            d_model: Model dimension
            ffn_hidden: Feed-forward hidden dimension
            n_head: Number of attention heads
            drop_prob: Dropout probability
        """
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head, drop_prob=drop_prob)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head, drop_prob=drop_prob)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(
        self, 
        dec: torch.Tensor, 
        enc: Optional[torch.Tensor], 
        trg_mask: Optional[torch.Tensor] = None, 
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through decoder layer.
        
        Args:
            dec: Decoder input
                Shape: (batch_size, tgt_len, d_model)
            enc: Encoder output (optional)
                Shape: (batch_size, src_len, d_model)
            trg_mask: Target mask (for self-attention)
                Shape: (batch_size, tgt_len, tgt_len) or broadcastable
            src_mask: Source mask (for cross-attention)
                Shape: (batch_size, tgt_len, src_len) or broadcastable
        
        Returns:
            output: Decoded output
                Shape: (batch_size, tgt_len, d_model)
        """
        # 1. Compute self attention
        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)
        
        # 2. Add and norm (residual connection + layer normalization)
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if enc is not None:
            # 3. Compute encoder-decoder attention
            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)
            
            # 4. Add and norm (residual connection + layer normalization)
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        # 5. Position-wise feed forward network
        _x = x
        x = self.ffn(x)
        
        # 6. Add and norm (residual connection + layer normalization)
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        
        return x