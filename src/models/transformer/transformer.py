"""
Transformer Model
Based on "Attention is All You Need" (Vaswani et al., 2017)
Compatible with BaseModel interface
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple

from ..base_model import BaseModel
from .encoder import Encoder
from .decoder import Decoder


class Transformer(BaseModel):
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
            config: Configuration dictionary
            src_vocab_size: Source vocabulary size
            tgt_vocab_size: Target vocabulary size
        """
        super(Transformer, self).__init__(config, src_vocab_size, tgt_vocab_size)
        
        # Model hyperparameters
        self.num_heads = config.get("model.num_heads", 8)
        self.num_layers = config.get("model.num_layers", 6)
        self.ff_dim = config.get("model.ff_dim", 2048)
        self.max_len = config.get("data.max_seq_len", 100)
        
        # Special token IDs
        self.src_pad_idx = config.get("data.pad_id", 0)
        self.tgt_pad_idx = config.get("data.pad_id", 0)
        self.tgt_sos_idx = config.get("data.sos_id", 1)
        self.tgt_eos_idx = config.get("data.eos_id", 2)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create encoder and decoder
        self.encoder = Encoder(
            enc_voc_size=src_vocab_size,
            max_len=self.max_len,
            d_model=self.embed_dim,
            ffn_hidden=self.ff_dim,
            n_head=self.num_heads,
            n_layers=self.num_layers,
            drop_prob=self.dropout_rate,
            device=self.device,
            padding_idx=self.src_pad_idx
        )
        
        self.decoder = Decoder(
            dec_voc_size=tgt_vocab_size,
            max_len=self.max_len,
            d_model=self.embed_dim,
            ffn_hidden=self.ff_dim,
            n_head=self.num_heads,
            n_layers=self.num_layers,
            drop_prob=self.dropout_rate,
            device=self.device,
            padding_idx=self.tgt_pad_idx
        )
        
        # Output projection
        self.output_projection = nn.Linear(self.embed_dim, tgt_vocab_size)
        
        # Tied embeddings flag
        self.tied_embeddings = config.get("model.tied_embeddings", False)
        if self.tied_embeddings:
            self._apply_tied_embeddings()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0.0)
                nn.init.constant_(module.weight, 1.0)
    
    def _apply_tied_embeddings(self):
        """Apply weight tying between target embedding and output projection."""
        if not hasattr(self.decoder, 'emb') or not hasattr(self, 'output_projection'):
            raise RuntimeError("Decoder must have 'emb' and output_projection must exist before applying tied embeddings")
        # Share weights: output_projection.weight will reference decoder embedding weight
        self.output_projection.weight = self.decoder.emb.tok_emb.weight
    
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
        mask = torch.tril(torch.ones(size, size, device=device, dtype=torch.bool))
        return mask
    
    @staticmethod
    def create_padding_mask(padding_mask_2d: torch.Tensor) -> torch.Tensor:
        """
        Create 3D attention mask from 2D padding mask.
        
        Args:
            padding_mask_2d: Padding mask (batch, seq_len) - boolean mask where True = valid token
            
        Returns:
            attention_mask_3d: Combined mask (batch, seq_len, seq_len)
        """
        # (B, L) -> (B, L, L)
        attention_mask_3d = padding_mask_2d.unsqueeze(2) & padding_mask_2d.unsqueeze(1)
        return attention_mask_3d.to(dtype=torch.bool)
    
    @staticmethod
    def create_cross_attention_mask(
        tgt_padding_mask: torch.Tensor,  # (B, T)
        src_padding_mask: torch.Tensor   # (B, L)
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
        if src_padding_mask.size(0) != batch_size:
            raise ValueError(
                f"Batch size mismatch: tgt_padding_mask has batch_size={batch_size}, "
                f"src_padding_mask has batch_size={src_padding_mask.size(0)}"
            )
        
        # Ensure boolean dtype
        tgt_padding_mask = tgt_padding_mask.to(dtype=torch.bool)
        src_padding_mask = src_padding_mask.to(dtype=torch.bool)
        
        # (B, T) -> (B, T, 1)
        tgt_expanded = tgt_padding_mask.unsqueeze(2)
        # (B, L) -> (B, 1, L)
        src_expanded = src_padding_mask.unsqueeze(1)
        
        # Broadcast: (B, T, 1) & (B, 1, L) -> (B, T, L)
        mask = tgt_expanded & src_expanded
        
        return mask
    
    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        Create source mask from source sequence.
        
        Args:
            src: Source sequence
                Shape: (batch_size, src_len)
        
        Returns:
            src_mask: Source mask
                Shape: (batch_size, 1, 1, src_len)
        """
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def make_trg_mask(self, trg: torch.Tensor) -> torch.Tensor:
        """
        Create target mask (look-ahead + padding) from target sequence.
        
        Args:
            trg: Target sequence
                Shape: (batch_size, tgt_len)
        
        Returns:
            trg_mask: Target mask
                Shape: (batch_size, 1, tgt_len, tgt_len)
        """
        # CRITICAL FIX: unsqueeze(2) not unsqueeze(3) to mask in Key dimension
        # Shape: (Batch, 1, 1, tgt_len) - "At any position i, don't attend to position j if j is padding"
        trg_pad_mask = (trg != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len, device=trg.device)).bool()  # (L, L)
        # Broadcasting: (B, 1, 1, L) & (L, L) -> (B, 1, L, L)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask
    
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
            src_seq: Source sequence
                Shape: (batch_size, src_len)
            tgt_seq: Target sequence
                Shape: (batch_size, tgt_len)
            src_mask: Source mask (optional, will be created if None)
            tgt_mask: Target mask (optional, will be created if None)
        
        Returns:
            output: Model predictions
                Shape: (batch_size, tgt_len, tgt_vocab_size)
        """
        # Create masks (exactly like reference implementation)
        if src_mask is None:
            src_mask = self.make_src_mask(src_seq)
        elif src_mask.dim() == 2:
            # Convert 2D mask to 4D format: (B, L) -> (B, 1, 1, L)
            src_mask = src_mask.to(dtype=torch.bool).unsqueeze(1).unsqueeze(2)
        elif src_mask.dim() == 4:
            # Ensure bool dtype
            src_mask = src_mask.to(dtype=torch.bool)
        
        if tgt_mask is None:
            tgt_mask = self.make_trg_mask(tgt_seq)
        elif tgt_mask.dim() == 2:
            # Convert 2D mask to 4D format using make_trg_mask logic
            tgt_mask = self.make_trg_mask(tgt_seq)  # Recreate from sequence to ensure correct format
        elif tgt_mask.dim() == 4:
            # Ensure bool dtype
            tgt_mask = tgt_mask.to(dtype=torch.bool)
        
        # Encode
        enc_src = self.encoder(src_seq, src_mask)
        
        # Decode
        d_out = self.decoder(tgt_seq, enc_src, tgt_mask, src_mask)
        
        # Output projection 
        output = self.output_projection(d_out)
        return output
    
    def encode(
        self,
        src_seq: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode source sequence.
        
        Args:
            src_seq: Source sequence
                Shape: (batch_size, src_len)
            src_mask: Source padding mask (2D) or 3D mask
        
        Returns:
            encoder_output: Encoded output
                Shape: (batch_size, src_len, embed_dim)
        """
        device = src_seq.device
        
        # Prepare source mask
        if src_mask is None:
            src_mask = self.make_src_mask(src_seq)
        else:
            if src_mask.dim() == 2:
                src_mask = src_mask.to(device=device, dtype=torch.bool)
                src_mask_3d = self.create_padding_mask(src_mask)
                src_mask = src_mask_3d.unsqueeze(1)
            elif src_mask.dim() == 3:
                src_mask = src_mask.to(device=device, dtype=torch.bool)
                src_mask = src_mask.unsqueeze(1)
            else:
                src_mask = src_mask.to(device=device, dtype=torch.bool)
        
        encoder_output = self.encoder(src_seq, src_mask)
        return encoder_output
    
    def decode_step(
        self,
        tgt_token: torch.Tensor,
        encoder_output: torch.Tensor,
        past_key_values: Optional[Dict[str, Any]] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Decode a single step with proper autoregressive decoding.
        
        Args:
            tgt_token: Current target token (batch_size,)
            encoder_output: Encoder output (batch_size, src_len, embed_dim)
            past_key_values: Cached decoder states from previous steps
                Contains 'decoder_states' tensor and 'step' int
            src_mask: Source padding mask (batch_size, src_len)
        
        Returns:
            logits: Next token logits (batch_size, tgt_vocab_size)
            past_key_values: Updated decoder states for next step
        """
        batch_size = tgt_token.size(0)
        device = tgt_token.device
        
        # Ensure src_mask is correct type
        if src_mask is not None:
            src_mask = src_mask.to(device=device, dtype=torch.bool)
            if src_mask.dim() == 2:
                src_mask_3d = self.create_padding_mask(src_mask)
                src_mask = src_mask_3d.unsqueeze(1)
            elif src_mask.dim() == 3:
                src_mask = src_mask.unsqueeze(1)
        
        # Initialize or retrieve past states
        if past_key_values is None:
            past_key_values = {'decoder_states': None, 'step': 0}
            current_step = 0
        else:
            current_step = past_key_values['step']
        
        # Embed current token with positional encoding offset
        tgt_embedded = self.decoder.emb(tgt_token.unsqueeze(1), offset=current_step)
        
        # Accumulate decoder states for self-attention
        if past_key_values['decoder_states'] is not None:
            decoder_input = torch.cat([past_key_values['decoder_states'], tgt_embedded], dim=1)
        else:
            decoder_input = tgt_embedded
        
        seq_len = decoder_input.size(1)
        
        # Create self-attention mask (look-ahead)
        look_ahead_mask = self.create_look_ahead_mask(seq_len, device=device)
        trg_mask = look_ahead_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
        
        # Create cross-attention mask
        if src_mask is not None:
            # If src_mask is 2D: (B, L) -> create cross mask (B, T, L)
            if src_mask.dim() == 2:
                src_mask_2d = src_mask.to(device=device, dtype=torch.bool)
                # (B, 1, L) -> expand to (B, T, L)
                src_mask_cross = src_mask_2d.unsqueeze(1).expand(batch_size, seq_len, -1)
            elif src_mask.dim() == 4:
                # (B, 1, L, L) -> extract key mask and expand: (B, T, L)
                # Take first query position's key mask
                src_mask_key = src_mask[:, 0, 0, :]  # (B, L)
                src_mask_cross = src_mask_key.unsqueeze(1).expand(batch_size, seq_len, -1)
            else:
                src_mask_cross = None
        else:
            src_mask_cross = None
        
        # Decode through all layers
        decoder_output = decoder_input
        for layer in self.decoder.layers:
            decoder_output = layer(
                decoder_output,
                encoder_output,
                trg_mask=trg_mask,
                src_mask=src_mask_cross
            )
        
        # Get output for the last position
        current_output = decoder_output[:, -1:, :]
        
        # Project to vocabulary using output_projection
        logits = self.output_projection(current_output)
        logits = logits.squeeze(dim=1)
        
        # Update past_key_values
        past_key_values['decoder_states'] = decoder_input
        past_key_values['step'] = current_step + 1
        
        return logits, past_key_values