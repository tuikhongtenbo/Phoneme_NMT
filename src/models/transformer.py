"""
Transformer model implementation
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Any, Optional

from .base_model import BaseModel
from .attention.multi_head import MultiHeadAttention


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for Transformer.
    
    References:
        Vaswani et al. (2017) "Attention is All You Need" NIPS 2017
    """
    
    def __init__(self, embed_dim: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize Positional Encoding.
        
        Args:
            embed_dim (int): Embedding dimension -> must be even
            max_len (int): Maximum sequence length
            dropout (float): Dropout rate
        """
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x (Tensor): Input embeddings
                Shape: (batch_size, seq_len, embed_dim)
        
        Returns:
            output (Tensor): Positionally encoded embeddings
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer.
    
    Single layer of the transformer encoder consisting of:
    - Multi-head self-attention
    - Feed-forward network
    - Layer normalization and residual connections
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1
    ):
        """
        Initialize Transformer Encoder Layer.
        
        Args:
            embed_dim (int): Model dimension
            num_heads (int): Number of attention heads
            ff_dim (int): Feed-forward hidden dimension
            dropout (float): Dropout rate
        """
        super(TransformerEncoderLayer, self).__init__()
        
        # Multi-head self-attention
        self.self_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through encoder layer.
        
        Args:
            x (Tensor): Input
                Shape: (batch_size, seq_len, embed_dim)
            mask (Tensor, optional): Attention mask
        
        Returns:
            output (Tensor): Encoded output
        """
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class TransformerDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer.
    
    Single layer of the transformer decoder consisting of:
    - Masked multi-head self-attention
    - Multi-head encoder-decoder attention
    - Feed-forward network
    - Layer normalization and residual connections
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1
    ):
        """
        Initialize Transformer Decoder Layer.
        
        Args:
            embed_dim (int): Model dimension
            num_heads (int): Number of attention heads
            ff_dim (int): Feed-forward hidden dimension
            dropout (float): Dropout rate
        """
        super(TransformerDecoderLayer, self).__init__()
        
        # Masked self-attention
        self.self_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        # Encoder-decoder attention
        self.cross_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through decoder layer.
        
        Args:
            x (Tensor): Decoder input
                Shape: (batch_size, tgt_len, embed_dim)
            encoder_output (Tensor): Encoder output
                Shape: (batch_size, src_len, embed_dim)
            self_mask (Tensor, optional): Masked self-attention mask
            cross_mask (Tensor, optional): Cross-attention mask
        
        Returns:
            output (Tensor): Decoded output
        """
        # Masked self-attention with residual connection and layer norm
        attn_output = self.self_attention(x, x, x, self_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention with residual connection and layer norm
        cross_attn_output = self.cross_attention(x, encoder_output, encoder_output, cross_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm3(x + ff_output)
        
        return x


class TransformerModel(BaseModel):
    """
    Transformer-based Neural Machine Translation Model.
    
    Architecture:
        - Encoder: Stack of transformer encoder layers
        - Decoder: Stack of transformer decoder layers
        - Multi-head attention mechanisms
        - Positional encoding
    
    References:
        Vaswani et al. (2017) "Attention is All You Need" NIPS 2017
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        src_vocab_size: int,
        tgt_vocab_size: int
    ):
        """
        Initialize Transformer Model.
        
        Args:
            config (Dict): Configuration dictionary
            src_vocab_size (int): Source vocabulary size
            tgt_vocab_size (int): Target vocabulary size
        """
        super(TransformerModel, self).__init__(config, src_vocab_size, tgt_vocab_size)
        
        # Model hyperparameters
        self.num_heads = config.get("model.num_heads", 8)
        self.num_layers = config.get("model.num_layers", 6)
        self.ff_dim = config.get("model.ff_dim", 2048)
        
        # Create encoder
        encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout=self.dropout_rate
            )
            for _ in range(self.num_layers)
        ])
        self.encoder_layers = encoder_layers
        
        # Create decoder
        decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout=self.dropout_rate
            )
            for _ in range(self.num_layers)
        ])
        self.decoder_layers = decoder_layers
        
        # Positional encoding
        self.src_pos_encoding = PositionalEncoding(self.embed_dim, dropout=self.dropout_rate)
        self.tgt_pos_encoding = PositionalEncoding(self.embed_dim, dropout=self.dropout_rate)
    
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
            tgt_mask (Tensor, optional): Target look-ahead mask + padding mask
        
        Returns:
            output (Tensor): Model predictions
                Shape: (batch_size, tgt_len, tgt_vocab_size)
        """
        # Source embeddings with positional encoding
        src_embedded = self.src_embedding(src_seq)
        src_embedded = self.src_pos_encoding(src_embedded)
        
        # Encode
        encoder_output = src_embedded
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, src_mask)
        
        # Target embeddings with positional encoding
        tgt_embedded = self.tgt_embedding(tgt_seq)
        tgt_embedded = self.tgt_pos_encoding(tgt_embedded)
        
        # Decode
        decoder_output = tgt_embedded
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output, tgt_mask, src_mask)
        
        # Project to vocabulary
        output = self.output_projection(decoder_output)
        
        return output
    
    def encode(
        self,
        src_seq: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode source sequence.
        
        Args:
            src_seq (Tensor): Source sequence
            src_mask (Tensor, optional): Source padding mask
        
        Returns:
            encoder_output (Tensor): Encoded output
        """
        src_embedded = self.src_embedding(src_seq)
        src_embedded = self.src_pos_encoding(src_embedded)
        
        encoder_output = src_embedded
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, src_mask)
        
        return encoder_output
    
    def decode_step(
        self,
        tgt_token: torch.Tensor,
        encoder_output: torch.Tensor,
        past_key_values: Optional[Any] = None
    ) -> tuple[torch.Tensor, Any]:
        """
        Decode a single step (simplified for greedy decoding).
        
        Args:
            tgt_token (Tensor): Current target token
            encoder_output (Tensor): Encoder output
            past_key_values (Any, optional): Not used in basic implementation
        
        Returns:
            logits (Tensor): Next token logits
            past_key_values (Any): Not used
        """
        
        # Embed and add positional encoding 
        tgt_embedded = self.tgt_embedding(tgt_token.unsqueeze(1))
        tgt_embedded = self.tgt_pos_encoding(tgt_embedded)
        
        # Single layer decode 
        decoder_output = tgt_embedded
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output)
        
        # Project to vocabulary
        logits = self.output_projection(decoder_output).squeeze(1)
        
        return logits, None