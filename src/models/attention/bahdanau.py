"""
Bahdanau Attention Mechanism
Implements the additive attention mechanism proposed by Bahdanau et al. (2015)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BahdanauAttention(nn.Module):
    """
    Bahdanau Attention (Additive Attention) Mechanism.
    
    This attention mechanism computes alignment scores using a FFN
    and then applies a softmax to get attention weights.
    
    Formula:
        energy = v^T * tanh(W1 * decoder_hidden + W2 * encoder_output)
        attention_weights = softmax(energy)
        context_vector = sum(attention_weights * encoder_output)
    
    References:
        Bahdanau et al. (2015) "Neural Machine Translation by Jointly Learning
        to Align and Translate" ICLR 2015
    """
    
    def __init__(
        self,
        hidden_dim: int,
        alignment_dim: Optional[int] = None
    ):
        """
        Initialize Bahdanau Attention.
        
        Args:
            hidden_dim (int): Dimension of hidden states
            alignment_dim (int, optional): Dimension for alignment computation
                If None, defaults to hidden_dim
        """
        super(BahdanauAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.alignment_dim = alignment_dim if alignment_dim is not None else hidden_dim
        
        # Learnable parameters for alignment computation
        self.W1 = nn.Linear(self.hidden_dim, self.alignment_dim, bias=False)
        self.W2 = nn.Linear(self.hidden_dim, self.alignment_dim, bias=False)
        self.v = nn.Linear(self.alignment_dim, 1, bias=False)
    
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
        
        # Expand decoder_hidden to match encoder_outputs length
        # (batch_size, hidden_dim) -> (batch_size, src_seq_len, hidden_dim)
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1).expand_as(encoder_outputs)
        
        # Compute alignment scores (energy)
        # (batch_size, src_seq_len, hidden_dim)
        alignment_scores = torch.tanh(
            self.W1(decoder_hidden_expanded) + self.W2(encoder_outputs)
        )
        
        # Compute energy scores
        # (batch_size, src_seq_len, alignment_dim) -> (batch_size, src_seq_len, 1) -> (batch_size, src_seq_len)
        energy = self.v(alignment_scores).squeeze(-1)  # (batch_size, src_seq_len)
        
        # Apply mask if provided
        if mask is not None:
            energy.masked_fill_(mask == 0, float(-1e9))
        
        # Compute attention weights using softmax
        attention_weights = F.softmax(energy, dim=1)  # (batch_size, src_seq_len)
        
        # Compute context vector as weighted sum of encoder outputs
        # (batch_size, src_seq_len, 1) * (batch_size, src_seq_len, hidden_dim)
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch_size, 1, src_seq_len)
            encoder_outputs  # (batch_size, src_seq_len, hidden_dim)
        ).squeeze(1)  # (batch_size, hidden_dim)
        
        return context_vector, attention_weights