"""
Luong Attention Mechanism
Implements the multiplicative attention mechanism proposed by Luong et al. (2015)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LuongAttention(nn.Module):
    """
    Luong Attention Mechanism.
    
    This attention mechanism computes alignment scores using dot product or
    general multiplicative attention.
    
    Attention types:
        - 'general': Uses learned matrix multiplication
        - 'concat': Concatenates decoder and encoder hidden states
        - 'dot': Simple dot product
    
    References:
        Luong et al. (2015) "Effective Approaches to Attention-based 
        Neural Machine Translation" EMNLP 2015
    """
    
    def __init__(
        self,
        hidden_dim: int,
        attention_type: str = 'general'
    ):
        """
        Initialize Luong Attention.
        
        Args:
            hidden_dim (int): Dimension of hidden states
            attention_type (str): Type of attention mechanism
                Options: 'general', 'concat', 'dot'
        """
        super(LuongAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.attention_type = attention_type
        
        if attention_type == 'general':
            # General attention: learned linear transformation
            self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        elif attention_type == 'concat':
            # Concat attention: concatenate and project
            self.linear = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
            self.v = nn.Linear(hidden_dim, 1, bias=False)
        elif attention_type == 'dot':
            # Dot product: no parameters needed
            self.linear = None
        else:
            raise ValueError(
                f"Attention type '{attention_type}' not supported. "
                "Choose from 'general', 'concat', or 'dot'"
            )
    
    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention weights and context vector.
        
        Args:
            decoder_hidden (Tensor): Current decoder hidden state
                Shape: (batch_size, hidden_dim)
            encoder_outputs (Tensor): All encoder hidden states
                Shape: (batch_size, src_seq_len, hidden_dim)
            mask (Tensor, optional): Padding mask for source sequence
                Shape: (batch_size, src_seq_len)
                1 for valid tokens, 0 for padding
        
        Returns:
            context_vector (Tensor): Weighted sum of encoder outputs
                Shape: (batch_size, hidden_dim)
            attention_weights (Tensor): Attention distribution over source positions
                Shape: (batch_size, src_seq_len)
        """
        # decoder_hidden: (batch_size, hidden_dim)
        # encoder_outputs: (batch_size, src_seq_len, hidden_dim)
        
        if self.attention_type == 'dot':
            # Dot product attention
            # (batch_size, hidden_dim) @ (batch_size, hidden_dim, src_seq_len) -> (batch_size, src_seq_len)
            energy = torch.bmm(
                decoder_hidden.unsqueeze(1),  # (batch_size, 1, hidden_dim)
                encoder_outputs.transpose(1, 2)  # (batch_size, hidden_dim, src_seq_len)
            ).squeeze(1)
        
        elif self.attention_type == 'general':
            # General attention
            encoder_outputs_transformed = self.linear(encoder_outputs)
            # (batch_size, 1, hidden_dim) @ (batch_size, hidden_dim, src_seq_len) -> (batch_size, src_seq_len)
            energy = torch.bmm(
                decoder_hidden.unsqueeze(1),  # (batch_size, 1, hidden_dim)
                encoder_outputs_transformed.transpose(1, 2)  # (batch_size, hidden_dim, src_seq_len)
            ).squeeze(1)
        
        elif self.attention_type == 'concat':
            # Concat attention
            decoder_hidden_expanded = decoder_hidden.unsqueeze(1).expand_as(encoder_outputs)
            # Concatenate and project
            concat_features = torch.cat([decoder_hidden_expanded, encoder_outputs], dim=2)
            energy = self.v(torch.tanh(self.linear(concat_features))).squeeze(-1)
        
        # Apply mask if provided
        if mask is not None:
            energy.masked_fill_(mask == 0, float(-1e9))
        
        # Compute attention weights using softmax
        attention_weights = F.softmax(energy, dim=1)  # (batch_size, src_seq_len)
        
        # Compute context vector as weighted sum of encoder outputs
        # (batch_size, 1, src_seq_len) @ (batch_size, src_seq_len, hidden_dim) -> (batch_size, hidden_dim)
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch_size, 1, src_seq_len)
            encoder_outputs  # (batch_size, src_seq_len, hidden_dim)
        ).squeeze(1)  # (batch_size, hidden_dim)
        
        return context_vector, attention_weights