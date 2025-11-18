"""
Transformer Decoder
Based on "Attention is All You Need" (Vaswani et al., 2017)
"""

import torch
import torch.nn as nn
from typing import Optional

from .blocks.decoder_layer import DecoderLayer
from .embedding.transformer_embedding import TransformerEmbedding


class Decoder(nn.Module):
    """
    Transformer Decoder.
    
    Stack of decoder layers with embedding and output projection.
    """
    
    def __init__(
        self, 
        dec_voc_size: int, 
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
        Initialize Decoder.
        
        Args:
            dec_voc_size: Target vocabulary size
            max_len: Maximum sequence length
            d_model: Model dimension
            ffn_hidden: Feed-forward hidden dimension
            n_head: Number of attention heads
            n_layers: Number of decoder layers
            drop_prob: Dropout probability
            device: Device to create model on
            padding_idx: Index of padding token
        """
        super().__init__()
        self.emb = TransformerEmbedding(
            vocab_size=dec_voc_size,
            d_model=d_model,
            max_len=max_len,
            drop_prob=drop_prob,
            padding_idx=padding_idx,
            device=device
        )

        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=d_model,
                ffn_hidden=ffn_hidden,
                n_head=n_head,
                drop_prob=drop_prob
            )
            for _ in range(n_layers)
        ])

    def forward(
        self, 
        trg: torch.Tensor, 
        enc_src: Optional[torch.Tensor], 
        trg_mask: Optional[torch.Tensor] = None, 
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            trg: Target sequence
                Shape: (batch_size, tgt_len)
            enc_src: Encoder output
                Shape: (batch_size, src_len, d_model)
            trg_mask: Target mask (for self-attention)
                Shape: (batch_size, tgt_len, tgt_len) or broadcastable
            src_mask: Source mask (for cross-attention)
                Shape: (batch_size, tgt_len, src_len) or broadcastable
        
        Returns:
            Decoder output (hidden states)
                Shape: (batch_size, tgt_len, d_model)
        """
        trg = self.emb(trg)

        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        return trg