"""
Transformer Embedding: Token Embedding + Positional Encoding
Based on "Attention is All You Need" (Vaswani et al., 2017)
"""

import torch
import torch.nn as nn
from typing import Optional

from .positional_encoding import PositionalEncoding
from .token_embeddings import TokenEmbedding


class TransformerEmbedding(nn.Module):
    """
    Token embedding + positional encoding (sinusoid).
    
    Positional encoding provides positional information to the network
    since Transformer has no inherent notion of sequence order.
    """

    def __init__(
        self, 
        vocab_size: int, 
        d_model: int, 
        max_len: int = 5000, 
        drop_prob: float = 0.1,
        padding_idx: int = 0,
        device: Optional[torch.device] = None
    ):
        """
        Initialize Transformer Embedding.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Dimensions of model
            max_len: Maximum sequence length
            drop_prob: Dropout probability
            padding_idx: Index of padding token
            device: Device to create positional encoding on
        """
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = PositionalEncoding(d_model, max_len, device=device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Forward pass through transformer embedding.
        
        Args:
            x: Input token indices
                Shape: (batch_size, seq_len)
            offset: Position offset for incremental decoding (default: 0)
        
        Returns:
            Embedded output with positional encoding
                Shape: (batch_size, seq_len, d_model)
        """
        tok_emb = self.tok_emb(x)  # (batch_size, seq_len, d_model)
        pos_emb = self.pos_emb(x, offset=offset)  # (seq_len, d_model)
        
        # Add positional encoding (broadcast over batch)
        return self.drop_out(tok_emb + pos_emb)