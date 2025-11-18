"""
Token Embedding
Based on "Attention is All You Need" (Vaswani et al., 2017)
"""

import torch.nn as nn


class TokenEmbedding(nn.Embedding):
    """
    Token Embedding using torch.nn.Embedding.
    
    Provides dense representation of words using weighted matrix.
    """

    def __init__(self, vocab_size: int, d_model: int, padding_idx: int = 0):
        """
        Initialize Token Embedding.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Embedding dimension
            padding_idx: Index of padding token (default: 0)
        """
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=padding_idx)