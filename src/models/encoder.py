"""
LSTM Encoder Module for Vie-Eng NMT
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple


class LSTMEncoder(nn.Module):
    """
    LSTM Encoder Module for Vie-Eng NMT

    Architecture:
        - Input embedding layer
        - LSTM layers
        - Output: Hidden states and cell states
    References:
        Bahdanau et al. (2015) "Neural Machine Translation by Jointly Learning
        to Align and Translate" ICLR 2015
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
            bidirectional (bool): Whether to use bidirectional LSTM
        """
        super(LSTMEncoder, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Input embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self, 
        src_seq: torch.Tensor, 
        src_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor,Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the encoder.

        Args:
            src_seq (Tensor): Source sequence indices
                Shape: (batch_size, src_len)
            src_lengths (Tensor, optional): Actual lengths of the source sequences
                Shape: (batch_size,)
        Returns:
            encoder_outputs (Tensor): All hidden states from the LSTM 
                Shape: (batch_size, src_len, hidden_dim * num_directions)
            (hidden, cell) (Tuple): Final hidden and cell states 
                hidden Shape: (num_layers * num_directions, batch_size, hidden_dim)
                cell Shape: (num_layers * num_directions, batch_size, hidden_dim)
        """
        # Input embedding
        embedded = self.embedding(src_seq)
        embedded = self.dropout_layer(embedded)

        # Pack sequence for efficient processing
        if src_lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, src_lengths, batch_first=True, enforce_sorted=False
            )

        # LSTM forward pass
        encoder_outputs, (hidden, cell) = self.lstm(embedded)

        # Unpack padded sequence
        if src_lengths is not None:
            encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(encoder_outputs, batch_first=True)

        return encoder_outputs, (hidden, cell)