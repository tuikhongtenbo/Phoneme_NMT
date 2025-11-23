"""
LSTM + Bahdanau Attention model implementation
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple

from ..base_model import BaseModel
from .encoder import LSTMEncoder
from ..attention.bahdanau import BahdanauAttention


class LSTMDecoder(nn.Module):
    """
    LSTM Decoder with Bahdanau Attention.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize LSTM Decoder.
        
        Args:
            vocab_size (int): Size of target vocabulary
            embed_dim (int): Dimension of word embeddings
            hidden_dim (int): Dimension of LSTM hidden states
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout rate
        """
        super(LSTMDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Attention mechanism
        self.attention = BahdanauAttention(hidden_dim)
        
        # LSTM layer - input is concatenation of embedding and context
        self.lstm = nn.LSTM(
            input_size=embed_dim + hidden_dim,  # embedding + context vector
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(
        self,
        tgt_seq: torch.Tensor,
        encoder_outputs: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through decoder.
        
        Args:
            tgt_seq (Tensor): Target sequence indices
                Shape: (batch_size, tgt_seq_len)
            encoder_outputs (Tensor): Encoder outputs
                Shape: (batch_size, src_seq_len, hidden_dim)
            hidden (Tuple, optional): Initial hidden state
            mask (Tensor, optional): Padding mask
        
        Returns:
            decoder_outputs (Tensor): Decoder hidden states
                Shape: (batch_size, tgt_seq_len, hidden_dim)
            hidden (Tuple): Final hidden state
        """
        # Input embedding: (batch_size, tgt_seq_len, embed_dim)
        embedded = self.embedding(tgt_seq)
        embedded = self.dropout_layer(embedded)
        
        # Process each timestep
        decoder_outputs = []
        h_n = hidden[0] if hidden is not None else None
        c_n = hidden[1] if hidden is not None else None
        
        for t in range(tgt_seq.size(1)):
            # Current input token
            input_token = embedded[:, t:t+1, :]  # (batch_size, 1, embed_dim)
            
            # Use the last hidden state for attention
            if h_n is not None:
                # Get hidden state from last layer
                decoder_hidden = h_n[-1]  # (batch_size, hidden_dim)
                
                # Compute attention
                context_vector, _ = self.attention(
                    decoder_hidden, encoder_outputs, mask
                )
                context_vector = context_vector.unsqueeze(1)  # (batch_size, 1, hidden_dim)
            else:
                # First timestep: use zero context
                context_vector = torch.zeros(
                    embedded.size(0), 1, self.hidden_dim,
                    device=embedded.device
                )
            
            # Concatenate input and context
            lstm_input = torch.cat([input_token, context_vector], dim=2)
            
            # LSTM forward
            lstm_output, (h_n, c_n) = self.lstm(lstm_input, (h_n, c_n))
            
            decoder_outputs.append(lstm_output)
        
        # Concatenate all outputs
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        
        return decoder_outputs, (h_n, c_n)


class LSTMBahdanau(BaseModel):
    """
    LSTM-based Neural Machine Translation Model with Bahdanau Attention.
    
    Architecture:
        - Bidirectional LSTM Encoder
        - LSTM Decoder with Bahdanau Attention
        - Attention mechanism to align source and target sequences
    
    References:
        Bahdanau et al. (2015) "Neural Machine Translation by Jointly Learning
        to Align and Translate" ICLR 2015
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        src_vocab_size: int,
        tgt_vocab_size: int
    ):
        """
        Initialize LSTM Bahdanau Model.
        
        Args:
            config (Dict): Configuration dictionary
            src_vocab_size (int): Source vocabulary size
            tgt_vocab_size (int): Target vocabulary size
        """
        super(LSTMBahdanau, self).__init__(config, src_vocab_size, tgt_vocab_size)
        
        # Model hyperparameters
        self.hidden_dim = config.get("model.hidden_dim", 512)
        # Support separate encoder/decoder layers, fallback to num_layers
        self.encoder_layers = config.get("model.encoder_layers", config.get("model.num_layers", 2))
        self.decoder_layers = config.get("model.decoder_layers", config.get("model.num_layers", 2))
        self.dropout_rate = config.get("model.dropout", 0.1)
        
        # Initialize encoder and decoder
        self.encoder = LSTMEncoder(
            vocab_size=src_vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.encoder_layers,
            dropout=self.dropout_rate,
            bidirectional=False
        )
        
        self.decoder = LSTMDecoder(
            vocab_size=tgt_vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.decoder_layers,
            dropout=self.dropout_rate
        )
    
    def forward(
        self,
        src_seq: torch.Tensor,
        tgt_seq: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            src_seq (Tensor): Source sequence
                Shape: (batch_size, src_len)
            tgt_seq (Tensor): Target sequence
                Shape: (batch_size, tgt_len)
            src_mask (Tensor, optional): Source padding mask
            tgt_mask (Tensor, optional): Target padding mask
        
        Returns:
            output (Tensor): Model predictions
                Shape: (batch_size, tgt_len, tgt_vocab_size)
        """
        # Encode source sequence
        encoder_outputs, encoder_hidden = self.encoder(src_seq)
        
        # Decode target sequence
        decoder_outputs, _ = self.decoder(
            tgt_seq, encoder_outputs, encoder_hidden, src_mask
        )
        
        # Project to vocabulary size
        output = self.output_projection(decoder_outputs)
        
        return output
    
    def encode(
        self,
        src_seq: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode source sequence.
        
        Args:
            src_seq (Tensor): Source sequence
            src_mask (Tensor, optional): Source padding mask
        
        Returns:
            encoder_outputs (Tensor): Encoder outputs
            hidden (Tuple): Hidden states
        """
        encoder_outputs, encoder_hidden = self.encoder(src_seq)
        return encoder_outputs, encoder_hidden
    
    def decode_step(
        self,
        tgt_token: torch.Tensor,
        encoder_output: Any,
        past_key_values: Optional[Any] = None
    ) -> Tuple[torch.Tensor, Any]:
        """
        Decode a single step.
        
        Args:
            tgt_token (Tensor): Current target token
            encoder_output (Any): Encoder outputs
            past_key_values (Any, optional): Previous hidden states
        
        Returns:
            logits (Tensor): Next token logits
            hidden (Any): Updated hidden states
        """
        # Extract encoder outputs and hidden
        encoder_outputs, _ = encoder_output
        hidden = past_key_values if past_key_values is not None else None
        
        # Get hidden state for attention
        if hidden is not None:
            decoder_hidden = hidden[0][-1]  # Last layer hidden state
        else:
            # Use encoder's last hidden state
            decoder_hidden = encoder_outputs.mean(dim=1)  # Mean pooling
        
        # Compute attention
        context_vector, _ = self.decoder.attention(decoder_hidden, encoder_outputs)
        # context_vector shape: (batch_size, hidden_dim)
        context_vector = context_vector.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # Embed token
        embedded = self.decoder.embedding(tgt_token).unsqueeze(1)  # (batch_size, 1, embed_dim)
        
        # Concatenate and decode
        lstm_input = torch.cat([embedded, context_vector], dim=2)  # (batch_size, 1, embed_dim + hidden_dim)
        lstm_output, hidden = self.decoder.lstm(lstm_input, hidden)
        
        # Project to vocabulary
        logits = self.output_projection(lstm_output.squeeze(1))
        
        return logits, hidden