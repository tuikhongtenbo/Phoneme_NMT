"""
LSTM Seq2Seq model implementation without Attention
Based on: Sequence to Sequence Learning with Neural Networks
"""
import torch
import torch.nn as nn
import random
from typing import Dict, Any, Optional, Tuple

from ..base_model import BaseModel
from .encoder import LSTMEncoder
from .decoder import LSTMDecoder


class LSTMSeq2Seq(BaseModel):
    """
    Standard LSTM-based Neural Machine Translation Model, 
    without Attention (Method 1).
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        src_vocab_size: int,
        tgt_vocab_size: int
    ):
        """
        Initialize the LSTM Seq2Seq Model.
        """
        super().__init__(config, src_vocab_size, tgt_vocab_size)
        
        # Model hyperparameters
        self.hidden_dim = config.get("model.hidden_dim", 512)
        # Use simple uncoupled num_layers logic
        self.num_layers = config.get("model.num_layers", 2)
        self.dropout_rate = config.get("model.dropout", 0.1)
        
        # Initialize encoder
        self.encoder = LSTMEncoder(
            vocab_size=src_vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout_rate,
            bidirectional=False
        )
        
        # Initialize decoder
        self.decoder = LSTMDecoder(
            vocab_size=tgt_vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout_rate
        )
        
        # Final projection layer is removed, decoder.fc_out is used instead.
    
    def forward(
        self,
        src_seq: torch.Tensor,
        tgt_seq: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0  # Default 1.0 since our Trainer uses 100% TF
    ) -> torch.Tensor:
        """
        Forward pass through the model during training.
        
        Args:
            src_seq (Tensor): Source sequence
                Shape: (batch_size, src_len)
            tgt_seq (Tensor): Target sequence (Trainer passed tgt_input missing <eos>)
                Shape: (batch_size, tgt_len)
        
        Returns:
            outputs (Tensor): Model predictions / logits
                Shape: (batch_size, tgt_len, tgt_vocab_size)
        """
        batch_size = tgt_seq.shape[0]
        trg_length = tgt_seq.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_length, trg_vocab_size).to(tgt_seq.device)
        
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        _, (hidden, cell) = self.encoder(src_seq)
        
        # first input to the decoder is the <sos> tokens
        input = tgt_seq[:, 0]
        
        for t in range(0, trg_length):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            # place predictions in a tensor holding predictions for each token
            outputs[:, t] = output
            
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            # get the highest predicted token from our predictions
            top1 = output.argmax(1)
            
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            if t + 1 < trg_length:
                input = tgt_seq[:, t + 1] if teacher_force else top1
                
        return outputs
    
    def encode(
        self,
        src_seq: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode source sequence (Used mainly for inference).
        """
        encoder_outputs, (hidden, cell) = self.encoder(src_seq)
        return encoder_outputs, (hidden, cell)
    
    def decode_step(
        self,
        tgt_token: torch.Tensor,
        encoder_output: Any,
        past_key_values: Optional[Any] = None
    ) -> Tuple[torch.Tensor, Any]:
        """
        Decode a single step.
        """
        _, encoder_hidden = encoder_output
        hidden, cell = past_key_values if past_key_values is not None else encoder_hidden
        
        # Ensure correct shape (batch_size)
        if tgt_token.dim() == 2:
            tgt_token = tgt_token.squeeze(1)
            
        prediction, hidden, cell = self.decoder(tgt_token, hidden, cell)
        
        return prediction, (hidden, cell)
