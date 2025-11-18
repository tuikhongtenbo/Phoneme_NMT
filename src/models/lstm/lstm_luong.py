"""
LSTM + Luong Attention model implementation
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple

from ..base_model import BaseModel
from .encoder import LSTMEncoder
from ..attention.luong import LuongAttention


class LSTMDecoderLuong(nn.Module):
    """
    LSTM Decoder with Luong Attention.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
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
            attention_type (str): Type of Luong attention
        """
        super(LSTMDecoderLuong, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = LuongAttention(hidden_dim, attention_type)
        
        # Concatenation projection
        self.concat_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        
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
        # Input embedding
        # Shape: (batch_size, tgt_seq_len, embed_dim)
        embedded = self.embedding(tgt_seq)
        embedded = self.dropout_layer(embedded)
        
        # Process through LSTM
        lstm_output, hidden = self.lstm(embedded, hidden)
        
        # Apply attention and concat projection at each timestep
        decoder_outputs = []
        for t in range(lstm_output.size(1)):
            decoder_hidden = lstm_output[:, t, :]  # (batch_size, hidden_dim)
            
            # Compute attention
            context_vector, _ = self.attention(
                decoder_hidden, encoder_outputs, mask
            )
            
            # Concatenate decoder hidden and context
            concat_output = torch.cat([decoder_hidden, context_vector], dim=1)
            
            # Project to hidden_dim
            output = self.concat_projection(concat_output)
            decoder_outputs.append(output)
        
        # Concatenate all outputs
        decoder_outputs = torch.stack(decoder_outputs, dim=1)
        
        return decoder_outputs, hidden


class LSTMLuong(BaseModel):
    """
    LSTM-based Neural Machine Translation Model with Luong Attention.
    
    Architecture:
        - Bidirectional LSTM Encoder
        - LSTM Decoder with Luong Attention
        - Attention mechanism to align source and target sequences
    
    References:
        Luong et al. (2015) "Effective Approaches to Attention-based 
        Neural Machine Translation" EMNLP 2015
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        src_vocab_size: int,
        tgt_vocab_size: int
    ):
        """
        Initialize LSTM Luong Model.
        
        Args:
            config (Dict): Configuration dictionary
            src_vocab_size (int): Source vocabulary size
            tgt_vocab_size (int): Target vocabulary size
        """
        super(LSTMLuong, self).__init__(config, src_vocab_size, tgt_vocab_size)
        
        # Model hyperparameters
        self.hidden_dim = config.get("model.hidden_dim", 512)
        self.num_layers = config.get("model.num_layers", 2)
        self.dropout_rate = config.get("model.dropout", 0.1)
        attention_type = config.get("model.attention_type", "general")
        
        # Initialize encoder and decoder
        self.encoder = LSTMEncoder(
            vocab_size=src_vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout_rate,
            bidirectional=False
        )
        
        self.decoder = LSTMDecoderLuong(
            vocab_size=tgt_vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout_rate,
            attention_type=attention_type
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
        
        # Embed token
        embedded = self.decoder.embedding(tgt_token).unsqueeze(1)
        
        # LSTM forward
        lstm_output, hidden = self.decoder.lstm(embedded, hidden)
        decoder_hidden = lstm_output.squeeze(1)  # (batch_size, hidden_dim)
        
        # Compute attention
        context_vector, _ = self.decoder.attention(decoder_hidden, encoder_outputs)
        
        # Concatenate and project
        concat_output = torch.cat([decoder_hidden, context_vector], dim=1)
        output = self.decoder.concat_projection(concat_output)
        
        # Project to vocabulary
        logits = self.output_projection(output)
        
        return logits, hidden