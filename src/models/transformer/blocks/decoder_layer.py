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
        # 1. Normalize before self-attention 
        x_norm = self.norm1(dec)
        x_attn = self.self_attention(q=x_norm, k=x_norm, v=x_norm, mask=trg_mask)
        x = dec + self.dropout1(x_attn)

        if enc is not None:
            # 2. Normalize before encoder-decoder attention 
            x_norm = self.norm2(x)
            x_attn = self.enc_dec_attention(q=x_norm, k=enc, v=enc, mask=src_mask)
            x = x + self.dropout2(x_attn)

        # 3. Normalize before FFN 
        x_norm = self.norm3(x)
        x_ff = self.ffn(x_norm)
        x = x + self.dropout3(x_ff)
        
        return x