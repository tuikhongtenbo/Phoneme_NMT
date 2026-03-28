"""
LSTM Encoder Module
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple


class LSTMEncoder(nn.Module):
    """
    LSTM Encoder Module for Vie-Eng NMT
    
    Architecture:
        - Input embedding layer
        - Multi-layer LSTM
        - Output: Hidden states and cell states
    References:
        Sutskever et al. (2014) "Sequence to Sequence Learning with Neural Networks"
    """ 

    def __init__(
        self, 
        vocab_size: int, 
        embed_dim: int, 
        hidden_dim: int, 
        num_layers: int, 
        dropout: float, 
        bidirectional: bool = False
    ):
        """
        Initialize LSTM Encoder.

        Args:
            vocab_size (int): Size of the vocabulary
            embed_dim (int): Dimension of the word embeddings
            hidden_dim (int): Dimension of the hidden states
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout rate
            bidirectional (bool): Ignored, only for compatibility
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = num_layers
        
        # Input embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # LSTM layers
        self.rnn = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, 
        src: torch.Tensor, 
        src_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the encoder.

        Args:
            src (Tensor): Source sequence indices
                Shape: (batch_size, src_len)
            src_lengths (Tensor, optional): Actual lengths of the source sequences
                Shape: (batch_size,) - Kept for API compatibility
        Returns:
            outputs (Tensor): All hidden states from the LSTM 
            (hidden, cell) (Tuple): Final hidden and cell states 
        """
        # src = [batch size, src length]
        
        embedded = self.dropout(self.embedding(src))
        # embedded = [batch size, src length, embedding dim]
        
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs = [batch size, src length, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        
        # outputs are always from the top hidden layer
        return outputs, (hidden, cell)