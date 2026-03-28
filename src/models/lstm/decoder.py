"""
LSTM Decoder Module for Vie-Eng NMT
Based on: Sequence to Sequence Learning with Neural Networks (No Attention)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class LSTMDecoder(nn.Module):
    """
    LSTM Decoder without Attention.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_attention: bool = False,
        attention_type: str = 'general'
    ):
        """
        Initialize LSTM Decoder.
        
        Args:
            vocab_size (int): Size of target vocabulary
            embed_dim (int): Dimension of word embeddings
            hidden_dim (int): Dimension of LSTM hidden states
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout rate
            use_attention (bool): Ignored, only for compatibility
            attention_type (str): Ignored, only for compatibility
        """
        super().__init__()
        
        self.output_dim = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = num_layers
        
        # Input embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # LSTM layer
        self.rnn = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Linear layer for prediction
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        input: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through decoder for a single time step.
        
        Args:
            input (Tensor): Target token index for a single step
                Shape: (batch_size)
            hidden (Tensor): Previous hidden state
                Shape: (num_layers, batch_size, hidden_dim)
            cell (Tensor): Previous cell state
                Shape: (num_layers, batch_size, hidden_dim)
        
        Returns:
            prediction (Tensor): Decoder prediction
                Shape: (batch_size, output_dim)
            hidden (Tensor): Updated hidden state
            cell (Tensor): Updated cell state
        """
        # input = [batch size]
        input = input.unsqueeze(1)
        # input = [batch size, 1]
        
        embedded = self.dropout(self.embedding(input))
        # embedded = [batch size, 1, embedding dim]
        
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output = [batch size, 1, hidden dim]
        # hidden = [n layers, batch size, hidden dim]
        # cell = [n layers, batch size, hidden dim]
        
        prediction = self.fc_out(output.squeeze(1))
        # prediction = [batch size, output dim]
        
        return prediction, hidden, cell
