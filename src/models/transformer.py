"""
Transformer model implementation
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Any, Optional, Tuple

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
        
        # Ensure embed_dim is even for sin/cos pairs
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim must be even, got {embed_dim}")
        
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Generate div_term for sin/cos pairs (step by 2)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x (Tensor): Input embeddings
                Shape: (batch_size, seq_len, embed_dim)
            offset (int): Position offset for incremental decoding
        
        Returns:
            output (Tensor): Positionally encoded embeddings
        """
        seq_len = x.size(1)
        x = x + self.pe[:, offset:offset+seq_len, :]
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
                Shape: (batch_size, seq_len, seq_len)
        
        Returns:
            output (Tensor): Encoded output
        """
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
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
                Shape: (batch_size, tgt_len, tgt_len)
            cross_mask (Tensor, optional): Cross-attention mask
                Shape: (batch_size, tgt_len, src_len)
        
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
        x = self.norm3(x + self.dropout(ff_output))
        
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
        
        # Tied embeddings flag 
        self.tied_embeddings = config.get("model.tied_embeddings", False)
        
        # Apply tied embeddings after all modules are initialized
        if self.tied_embeddings:
            self._apply_tied_embeddings()
    
    def _apply_tied_embeddings(self):
        """
        Apply weight tying between target embedding and output projection.
        """
        if not hasattr(self, 'tgt_embedding') or not hasattr(self, 'output_projection'):
            raise RuntimeError(
                "tgt_embedding and output_projection must be initialized by BaseModel "
                "before applying tied embeddings. Ensure super().__init__() is called first."
            )
        
        # Share weights: output_projection.weight will reference tgt_embedding.weight
        self.output_projection.weight = self.tgt_embedding.weight
    
    @staticmethod
    def create_look_ahead_mask(size: int, device: torch.device = None) -> torch.Tensor:
        """
        Create look-ahead mask for decoder (prevents attending to future tokens).
        
        Args:
            size: Sequence length
            device: Device to create mask on
            
        Returns:
            mask: Boolean mask (size, size), True for positions that can attend
        """
        # Create lower triangular mask: (size, size)
        mask = torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)
        mask = ~mask  # Invert: True means can attend
        return mask
    
    @staticmethod
    def create_padding_mask(
        padding_mask_2d: torch.Tensor
    ) -> torch.Tensor:
        """
        Create 3D attention mask from 2D padding mask.
        
        Args:
            padding_mask_2d: Padding mask (batch, seq_len) - boolean mask where True = valid token
            
        Returns:
            combined_mask: Combined mask (batch, seq_len, seq_len)
        """
        # (B, L) -> (B, L, L)
        attention_mask_3d = padding_mask_2d.unsqueeze(2) & padding_mask_2d.unsqueeze(1)
        return attention_mask_3d.to(dtype=torch.bool)
    
    @staticmethod
    def create_cross_attention_mask(
        tgt_padding_mask: torch.Tensor, # (B, T)
        src_padding_mask: torch.Tensor  # (B, L)
    ) -> torch.Tensor:
        """
        Create 3D cross-attention mask.
        
        Args:
            tgt_padding_mask: Target padding mask (batch, tgt_len) - True = valid
            src_padding_mask: Source padding mask (batch, src_len) - True = valid
            
        Returns:
            combined_mask: Combined mask (batch, tgt_len, src_len)
        """
        # Ensure both masks are 2D and have same batch size
        if tgt_padding_mask.dim() != 2:
            raise ValueError(f"tgt_padding_mask must be 2D, got {tgt_padding_mask.dim()}D")
        if src_padding_mask.dim() != 2:
            raise ValueError(f"src_padding_mask must be 2D, got {src_padding_mask.dim()}D")
        
        batch_size = tgt_padding_mask.size(0)
        tgt_len = tgt_padding_mask.size(1)
        src_len = src_padding_mask.size(1)
        
        if src_padding_mask.size(0) != batch_size:
            raise ValueError(
                f"Batch size mismatch: tgt_padding_mask has batch_size={batch_size}, "
                f"src_padding_mask has batch_size={src_padding_mask.size(0)}"
            )
        
        # Ensure boolean dtype
        tgt_padding_mask = tgt_padding_mask.to(dtype=torch.bool)
        src_padding_mask = src_padding_mask.to(dtype=torch.bool)
        
        # (B, T) -> (B, T, 1)
        tgt_expanded = tgt_padding_mask.unsqueeze(2)  # (B, T, 1)
        # (B, L) -> (B, 1, L)
        src_expanded = src_padding_mask.unsqueeze(1)  # (B, 1, L)
        
        # Broadcast: (B, T, 1) & (B, 1, L) -> (B, T, L)
        mask = tgt_expanded & src_expanded
        
        return mask
    
    def forward(
        self,
        src_seq: torch.Tensor,
        tgt_seq: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None, # 2D: (B, L) - True = valid
        tgt_mask: Optional[torch.Tensor] = None  # 2D: (B, T) - True = valid
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            src_seq (Tensor): Source sequence
                Shape: (batch_size, src_len)
            tgt_seq (Tensor): Target sequence
                Shape: (batch_size, tgt_len)
            src_mask (Tensor, optional): Source padding mask (batch_size, src_len)
                True for valid tokens, False for padding
            tgt_mask (Tensor, optional): Target padding mask (batch_size, tgt_len)
                True for valid tokens, False for padding
        
        Returns:
            output (Tensor): Model predictions
                Shape: (batch_size, tgt_len, tgt_vocab_size)
        """
        device = src_seq.device
        
        # 1. Encoder mask (B, L, L)
        src_attention_mask = None
        if src_mask is not None:
            src_mask = src_mask.to(device=device, dtype=torch.bool)
            src_attention_mask = self.create_padding_mask(src_mask)

        # 2. Decoder self-attention mask (B, T, T)
        tgt_self_attention_mask = None
        if tgt_mask is not None:
            tgt_mask = tgt_mask.to(device=device, dtype=torch.bool)
            tgt_len = tgt_seq.size(1)
            
            # Handle both 2D and 3D tgt_mask for backward compatibility with trainer
            if tgt_mask.dim() == 3:
                # Already 3D combined mask (B, T, T) from trainer
                tgt_self_attention_mask = tgt_mask
            elif tgt_mask.dim() == 2:
                # 2D padding mask (B, T) 
                # (B, T, T)
                tgt_padding_mask_3d = self.create_padding_mask(tgt_mask)
                
                # (T, T)
                look_ahead_mask = self.create_look_ahead_mask(tgt_len, device=device)
                
                # (B, T, T) & (T, T) -> (B, T, T)
                tgt_self_attention_mask = tgt_padding_mask_3d & look_ahead_mask.unsqueeze(0)
            else:
                raise ValueError(f"tgt_mask must be 2D or 3D, got {tgt_mask.dim()}D")
            
        # 3. Decoder cross-attention mask (B, T, L)
        cross_attention_mask = None
        if src_mask is not None and tgt_mask is not None:
            # Handle both 2D and 3D tgt_mask for backward compatibility with trainer
            if tgt_mask.dim() == 3:
                tgt_mask_2d = tgt_mask[:, 0, :]  # (B, T) - mask for query position 0
            elif tgt_mask.dim() == 2:
                tgt_mask_2d = tgt_mask  # Already 2D (B, T)
            else:
                raise ValueError(f"tgt_mask must be 2D or 3D, got {tgt_mask.dim()}D")
            
            # Ensure tgt_mask_2d is 2D and has correct dtype/device
            tgt_mask_2d = tgt_mask_2d.to(device=device, dtype=torch.bool)
            if tgt_mask_2d.dim() != 2:
                raise ValueError(f"tgt_mask_2d must be 2D after processing, got {tgt_mask_2d.dim()}D")
            
            cross_attention_mask = self.create_cross_attention_mask(tgt_mask_2d, src_mask)
        
        # Source embeddings with positional encoding
        src_embedded = self.src_embedding(src_seq)
        src_embedded = self.src_pos_encoding(src_embedded) 
        
        # Encode
        encoder_output = src_embedded
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, src_attention_mask)
        
        # Target embeddings with positional encoding
        tgt_embedded = self.tgt_embedding(tgt_seq)
        tgt_embedded = self.tgt_pos_encoding(tgt_embedded) 
        
        # Decode
        decoder_output = tgt_embedded
        for layer in self.decoder_layers:
            decoder_output = layer(
                decoder_output, 
                encoder_output, 
                tgt_self_attention_mask, # (B, T, T)
                cross_attention_mask     # (B, T, L)
            )
        
        # Project to vocabulary
        output = self.output_projection(decoder_output)
        
        return output
    
    def encode(
        self,
        src_seq: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None # 2D: (B, L)
    ) -> torch.Tensor:
        """
        Encode source sequence.
        
        Args:
            src_seq (Tensor): Source sequence
            src_mask (Tensor, optional): Source padding mask (2D)
        
        Returns:
            encoder_output (Tensor): Encoded output
        """
        device = src_seq.device
        
        src_attention_mask = None
        if src_mask is not None:
            src_mask = src_mask.to(device=device, dtype=torch.bool)
            src_attention_mask = self.create_padding_mask(src_mask) # (B, L, L)
            
        src_embedded = self.src_embedding(src_seq)
        src_embedded = self.src_pos_encoding(src_embedded)
        
        encoder_output = src_embedded
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, src_attention_mask)
        
        return encoder_output
    
    def decode_step(
        self,
        tgt_token: torch.Tensor, # (B,)
        encoder_output: torch.Tensor, # (B, L, D)
        past_key_values: Optional[Dict[str, Any]] = None,
        src_mask: Optional[torch.Tensor] = None # 2D: (B, L)
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Decode a single step with proper autoregressive decoding.

        Args:
            tgt_token (Tensor): Current target token (batch_size,)
            encoder_output (Tensor): Encoder output (batch_size, src_len, embed_dim)
            past_key_values (Dict, optional): Cached decoder states from previous steps
                Contains:
                    - 'decoder_states': tensor (batch_size, past_len, embed_dim) or None
                    - 'step': int, current decoding step
            src_mask (Tensor, optional): Source padding mask (batch_size, src_len)
                True for valid tokens, False for padding
        
        Returns:
            logits (Tensor): Next token logits (batch_size, tgt_vocab_size)
            past_key_values (Dict): Updated decoder states for next step
        """
        batch_size = tgt_token.size(0)
        device = tgt_token.device
        
        # Ensure src_mask (2D) is correct type
        if src_mask is not None:
            src_mask = src_mask.to(device=device, dtype=torch.bool)
        
        # Initialize or retrieve past states
        if past_key_values is None:
            past_key_values = {'decoder_states': None, 'step': 0}
            current_step = 0
        else:
            current_step = past_key_values['step']
        
        # Embed current token: (B,) -> (B, 1, D)
        tgt_embedded = self.tgt_embedding(tgt_token.unsqueeze(1))
        
        # Add positional encoding with exact offset
        tgt_embedded = self.tgt_pos_encoding(tgt_embedded, offset=current_step)
        
        # Accumulate decoder states for self-attention
        if past_key_values['decoder_states'] is not None:
            # Concatenate past states with current token
            decoder_input = torch.cat([past_key_values['decoder_states'], tgt_embedded], dim=1)
        else:
            decoder_input = tgt_embedded # (B, 1, D)
        
        seq_len = decoder_input.size(1) # seq_len = current_step + 1
        
        # 1. Create self-attention mask (B, T, T)
        look_ahead_mask = self.create_look_ahead_mask(seq_len, device=device)
        self_mask = look_ahead_mask.unsqueeze(0).expand(batch_size, -1, -1) # (B, T, T)

        # 2. Create cross-attention mask (B, T, L)
        cross_mask = None
        if src_mask is not None:
            # (B, 1, L)
            cross_mask = src_mask.unsqueeze(1)
            # Expand to (B, T, L)
            cross_mask = cross_mask.expand(batch_size, seq_len, -1)
            
        # Decode through all layers
        decoder_output = decoder_input
        for layer in self.decoder_layers:
            decoder_output = layer(
                decoder_output, 
                encoder_output, 
                self_mask=self_mask,
                cross_mask=cross_mask
            )
        
        current_output = decoder_output[:, -1:, :]  # (B, 1, D)
        
        # Project to vocabulary
        logits = self.output_projection(current_output)  # (B, 1, V)
        logits = logits.squeeze(dim=1)  # (B, V)
        
        # past_key_values for next step
        past_key_values['decoder_states'] = decoder_input  
        past_key_values['step'] = current_step + 1
        
        return logits, past_key_values