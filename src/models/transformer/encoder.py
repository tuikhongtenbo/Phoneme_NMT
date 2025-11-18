"""
Transformer Encoder
Based on "Attention is All You Need" (Vaswani et al., 2017)
"""

import torch
import torch.nn as nn
from typing import Optional

from .blocks.encoder_layer import EncoderLayer
from .embedding.transformer_embedding import TransformerEmbedding


class Encoder(nn.Module):
    """
    Transformer Encoder.
    
    Stack of encoder layers with embedding.
    """

    def __init__(
        self, 
        enc_voc_size: int, 
        max_len: int, 
        d_model: int, 
        ffn_hidden: int, 
        n_head: int, 
        n_layers: int, 
        drop_prob: float, 
        device: Optional[torch.device] = None,
        padding_idx: int = 0
    ):
        """
        Initialize Encoder.
        
        Args:
            enc_voc_size: Source vocabulary size
            max_len: Maximum sequence length
            d_model: Model dimension
            ffn_hidden: Feed-forward hidden dimension
            n_head: Number of attention heads
            n_layers: Number of encoder layers
            drop_prob: Dropout probability
            device: Device to create model on
            padding_idx: Index of padding token
        """
        super().__init__()
        self.emb = TransformerEmbedding(
            vocab_size=enc_voc_size,
            d_model=d_model,
            max_len=max_len,
            drop_prob=drop_prob,
            padding_idx=padding_idx,
            device=device
        )

        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                ffn_hidden=ffn_hidden,
                n_head=n_head,
                drop_prob=drop_prob
            )
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through encoder.
        
        Args:
            x: Source sequence
                Shape: (batch_size, src_len)
            src_mask: Source mask
                Shape: (batch_size, src_len, src_len) or broadcastable
        
        Returns:
            Encoded output
                Shape: (batch_size, src_len, d_model)
        """
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x