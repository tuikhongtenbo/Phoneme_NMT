"""
Main training class for Neural Machine Translation models.
Handles training loop, optimization, checkpointing, and evaluation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple, List
import os
import math
from pathlib import Path
import json
from tqdm import tqdm

from src.utils.logger import setup_logger
from src.evaluation.evaluator import Evaluator


class Trainer:
    """
    Main trainer class for NMT models.
    
    Handles:
    - Training loop with teacher forcing
    - Loss calculation and optimization
    - Checkpoint saving/loading
    - Evaluation during training
    - Learning rate scheduling
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Any,
        train_loader: DataLoader,
        dev_loader: Optional[DataLoader] = None,
        logger: Optional[Any] = None,
        input_vocab: Optional[Any] = None,
        output_vocab: Optional[Any] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: NMT model (Transformer, LSTMBahdanau, or LSTMLuong)
            config: Configuration object
            train_loader: Training data loader
            dev_loader: Development data loader (optional)
            logger: Logger instance (optional)
            input_vocab: Source vocabulary object (for decoding)
            output_vocab: Target vocabulary object (for decoding)
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.logger = logger or setup_logger(name="Trainer")
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        
        # Device setup
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.logger.info(f"Using device: {self.device}")
        
        # Training hyperparameters
        self.batch_size = config.training.batch_size
        self.num_epochs = config.training.num_epochs
        self.learning_rate = config.training.learning_rate
        self.clip_grad_norm = config.training.clip_grad_norm
        self.eval_every = config.training.eval_every
        self.save_every = config.training.save_every
        
        # Special token IDs
        self.pad_id = config.data.pad_id
        self.sos_id = config.data.sos_id
        self.eos_id = config.data.eos_id
        
        # Target level (word or phoneme)
        self.target_level = config.data.target_level
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Loss function (ignore padding tokens)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_id, reduction='mean')
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_dev_loss = float('inf')
        self.best_dev_bleu = 0.0
        self.metric_for_best = 'bleu'  
        
        # Checkpoint directory
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Evaluator
        self.evaluator = Evaluator(metrics=['bleu', 'rouge_l', 'meteor'])
        
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config."""
        optimizer_type = self.config.training.optimizer.lower()
        
        if optimizer_type == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_type == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.01
            )
        elif optimizer_type == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                nesterov=True
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def _create_scheduler(self) -> Optional[Any]:
        """Create learning rate scheduler based on config."""
        scheduler_type = self.config.training.scheduler
        
        if scheduler_type is None:
            return None
        
        scheduler_type = scheduler_type.lower()
        
        if scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs,
                eta_min=1e-6
            )
        elif scheduler_type == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.num_epochs // 3,
                gamma=0.1
            )
        elif scheduler_type == "warmup_cosine":
            # Custom warmup + cosine annealing
            warmup_steps = self.config.training.warmup_steps or 4000
            total_steps = self.num_epochs * len(self.train_loader)
            return optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: min(1.0, step / warmup_steps) if step < warmup_steps
                else 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
            )
        else:
            self.logger.warning(f"Unknown scheduler type: {scheduler_type}. Using no scheduler.")
            return None
    
    def _create_padding_mask(self, seq: torch.Tensor, pad_id: int) -> torch.Tensor:
        """
        Create padding mask for sequence.
        
        Args:
            seq: Input sequence (batch_size, seq_len)
            pad_id: Padding token ID
            
        Returns:
            mask: Boolean mask (batch_size, seq_len), True for non-padding tokens
        """
        return (seq != pad_id)
    
    def _create_look_ahead_mask(self, size: int) -> torch.Tensor:
        """
        Create look-ahead mask for decoder (prevents attending to future tokens).
        
        Args:
            size: Sequence length
            
        Returns:
            mask: Boolean mask (size, size), True for positions that can attend
        """
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return ~mask  # Invert: True means can attend
    
    def _prepare_batch(self, src_seq: torch.Tensor, tgt_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare batch for training.
        
        Args:
            src_seq: Source sequence (batch_size, src_len)
            tgt_seq: Target sequence (batch_size, tgt_len) or (batch_size, tgt_len, 4) for phoneme
            
        Returns:
            src_seq: Source sequence on device
            tgt_input: Target input (teacher forcing) on device
            tgt_output: Target output (for loss calculation) on device
            masks: Tuple of (src_mask, tgt_mask)
        """
        # Move to device
        src_seq = src_seq.to(self.device)
        tgt_seq = tgt_seq.to(self.device)
        
        # Handle phoneme-level targets (2D -> flatten for loss)
        if self.target_level == 'phoneme' and len(tgt_seq.shape) == 3:
            tgt_seq = tgt_seq[:, :, 0]
        
        # Create masks
        src_mask = self._create_padding_mask(src_seq, self.pad_id)
        tgt_mask = self._create_padding_mask(tgt_seq, self.pad_id)
        
        # Teacher forcing: input is tgt_seq[:-1], output is tgt_seq[1:]
        tgt_input = tgt_seq[:, :-1]  # Remove last token
        tgt_output = tgt_seq[:, 1:]   # Remove first token (SOS)
        
        # Create look-ahead mask for transformer decoder
        if self.config.model.name == "transformer":
            tgt_len = tgt_input.size(1)
            look_ahead = self._create_look_ahead_mask(tgt_len).to(self.device)
            # Combine with padding mask
            tgt_padding_mask = tgt_mask[:, 1:].unsqueeze(1)  # (batch, 1, tgt_len-1)
            tgt_mask_combined = look_ahead.unsqueeze(0) & tgt_padding_mask  # (batch, tgt_len-1, tgt_len-1)
        else:
            tgt_mask_combined = tgt_mask[:, 1:]  # For LSTM, just padding mask
        
        return src_seq, tgt_input, tgt_output, (src_mask, tgt_mask_combined)
    
    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            batch: (src_seq, tgt_seq) tuple
            
        Returns:
            Dictionary with loss and other metrics
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        src_seq, tgt_seq = batch
        src_seq, tgt_input, tgt_output, (src_mask, tgt_mask) = self._prepare_batch(src_seq, tgt_seq)
        
        # Forward pass
        logits = self.model(src_seq, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
        
        # Calculate loss
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        
        # Optimizer step
        self.optimizer.step()
        
        # Update learning rate for step-based schedulers
        if self.scheduler is not None and isinstance(self.scheduler, optim.lr_scheduler.LambdaLR):
            self.scheduler.step()
        
        # Calculate metrics
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == tgt_output).float().mean()
            non_pad_mask = (tgt_output != self.pad_id)
            if non_pad_mask.sum() > 0:
                accuracy = ((predictions == tgt_output) & non_pad_mask).float().sum() / non_pad_mask.sum()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'lr': self.optimizer.param_groups[0]['lr']
        }
    
    def _decode_indices_to_text(self, indices: List[int], vocab) -> str:
        """Decode indices to text string."""
        special_tokens = {0, 1, 2, 3}  # PAD, SOS, EOS, UNK
        
        if hasattr(vocab, 'index2word'):
            tokens = [vocab.index2word.get(idx, '<UNK>') for idx in indices if idx not in special_tokens]
        elif hasattr(vocab, 'itos'):
            tokens = [vocab.itos.get(idx, '<UNK>') for idx in indices if idx not in special_tokens]
        else:
            tokens = [str(idx) for idx in indices if idx not in special_tokens]
        
        return ' '.join(tokens)
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            data_loader: Data loader for evaluation
            
        Returns:
            Dictionary with evaluation metrics (loss, perplexity, bleu)
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        all_predictions_text = []
        all_targets_text = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating", leave=False):
                src_seq, tgt_seq = batch
                src_seq, tgt_input, tgt_output, (src_mask, tgt_mask) = self._prepare_batch(src_seq, tgt_seq)
                
                # Forward pass
                logits = self.model(src_seq, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
                
                # Calculate loss
                loss = self.criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
                
                # Count non-padding tokens
                non_pad_mask = (tgt_output != self.pad_id)
                num_tokens = non_pad_mask.sum().item()
                
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
                
                # Get predictions (greedy decoding)
                predictions = logits.argmax(dim=-1)  # (batch_size, tgt_len-1)
                
                # Decode predictions and targets to text for BLEU calculation
                if self.output_vocab is not None:
                    for pred_seq, tgt_seq in zip(predictions, tgt_output):
                        # Remove padding tokens
                        pred_ids = pred_seq[tgt_seq != self.pad_id].cpu().tolist()
                        tgt_ids = tgt_seq[tgt_seq != self.pad_id].cpu().tolist()
                        
                        # Stop at EOS
                        if self.eos_id in pred_ids:
                            pred_ids = pred_ids[:pred_ids.index(self.eos_id)]
                        if self.eos_id in tgt_ids:
                            tgt_ids = tgt_ids[:tgt_ids.index(self.eos_id)]
                        
                        # Decode to text
                        pred_text = self._decode_indices_to_text(pred_ids, self.output_vocab)
                        tgt_text = self._decode_indices_to_text(tgt_ids, self.output_vocab)
                        
                        all_predictions_text.append(pred_text)
                        all_targets_text.append(tgt_text)
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        
        metrics = {
            'loss': avg_loss,
            'perplexity': torch.exp(torch.tensor(avg_loss)).item() if avg_loss > 0 else float('inf')
        }
        
        # Calculate BLEU if we have decoded text
        if all_predictions_text and self.output_vocab is not None:
            try:
                bleu_scores = self.evaluator.evaluate(all_targets_text, all_predictions_text)
                metrics.update(bleu_scores)
                # Use average of BLEU-1, BLEU-2, BLEU-3, BLEU-4 as main metric
                bleu_1 = bleu_scores.get('bleu_1', 0.0)
                bleu_2 = bleu_scores.get('bleu_2', 0.0)
                bleu_3 = bleu_scores.get('bleu_3', 0.0)
                bleu_4 = bleu_scores.get('bleu_4', 0.0)
                metrics['bleu'] = (bleu_1 + bleu_2 + bleu_3 + bleu_4) / 4.0
            except Exception as e:
                self.logger.warning(f"Could not calculate BLEU: {e}")
                metrics['bleu'] = 0.0
        else:
            metrics['bleu'] = 0.0
        
        return metrics
    
    
    def save_checkpoint(self, is_best: bool = False, suffix: str = ""):
        """
        Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
            suffix: Optional suffix for checkpoint filename
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_dev_loss': self.best_dev_loss,
            'best_dev_bleu': self.best_dev_bleu,
            'config': self.config.model_dump() if hasattr(self.config, 'model_dump') else (self.config.dict() if hasattr(self.config, 'dict') else str(self.config))
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch{self.current_epoch}_step{self.global_step}{suffix}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_dev_loss = checkpoint.get('best_dev_loss', float('inf'))
        self.best_dev_bleu = checkpoint.get('best_dev_bleu', 0.0)
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        self.logger.info(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")
    
    def train(self, resume_from: Optional[str] = None):
        """
        Main training loop.
        
        Args:
            resume_from: Optional path to checkpoint to resume from
        """
        # Load checkpoint if resuming
        if resume_from:
            self.load_checkpoint(resume_from)
        
        self.logger.info("=" * 80)
        self.logger.info("Starting Training")
        self.logger.info("=" * 80)
        self.logger.info(f"Model: {self.config.model.name}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Learning rate: {self.learning_rate}")
        self.logger.info(f"Total epochs: {self.num_epochs}")
        self.logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        if self.dev_loader:
            self.logger.info(f"Dev samples: {len(self.dev_loader.dataset)}")
        self.logger.info("=" * 80)
        
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            self.logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            self.logger.info("-" * 80)
            
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            num_batches = 0
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")
            for batch in progress_bar:
                metrics = self.train_step(batch)
                
                epoch_loss += metrics['loss']
                epoch_accuracy += metrics['accuracy']
                num_batches += 1
                self.global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'acc': f"{metrics['accuracy']:.4f}",
                    'lr': f"{metrics['lr']:.2e}"
                })
                
                # Evaluation
                if self.dev_loader and self.global_step % self.eval_every == 0:
                    self.logger.info(f"\nEvaluating at step {self.global_step}...")
                    dev_metrics = self.evaluate(self.dev_loader)
                    self.logger.info(f"Dev Loss: {dev_metrics['loss']:.4f}, "
                                   f"Perplexity: {dev_metrics['perplexity']:.2f}")
                    if 'bleu' in dev_metrics:
                        self.logger.info(f"Dev BLEU: {dev_metrics['bleu']:.4f}")
                    
                    # Save if best (based on BLEU)
                    if self.metric_for_best == 'bleu':
                        is_best = dev_metrics.get('bleu', 0.0) > self.best_dev_bleu
                        if is_best:
                            self.best_dev_bleu = dev_metrics.get('bleu', 0.0)
                            self.logger.info(f"New best dev BLEU: {self.best_dev_bleu:.4f}")
                    else:
                        is_best = dev_metrics['loss'] < self.best_dev_loss
                        if is_best:
                            self.best_dev_loss = dev_metrics['loss']
                            self.logger.info(f"New best dev loss: {self.best_dev_loss:.4f}")
                
                # Save checkpoint
                if self.global_step % self.save_every == 0:
                    self.save_checkpoint(is_best=False, suffix=f"_step{self.global_step}")
            
            # Epoch summary
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            avg_accuracy = epoch_accuracy / num_batches if num_batches > 0 else 0.0
            
            self.logger.info(f"\nEpoch {epoch + 1} Summary:")
            self.logger.info(f"  Train Loss: {avg_loss:.4f}")
            self.logger.info(f"  Train Accuracy: {avg_accuracy:.4f}")
            
            # End-of-epoch evaluation
            if self.dev_loader:
                self.logger.info("Evaluating on dev set...")
                dev_metrics = self.evaluate(self.dev_loader)
                self.logger.info(f"  Dev Loss: {dev_metrics['loss']:.4f}")
                self.logger.info(f"  Dev Perplexity: {dev_metrics['perplexity']:.2f}")
                if 'bleu' in dev_metrics:
                    self.logger.info(f"  Dev BLEU: {dev_metrics['bleu']:.4f}")
                
                # Save if best (based on BLEU)
                if self.metric_for_best == 'bleu':
                    is_best = dev_metrics.get('bleu', 0.0) > self.best_dev_bleu
                    if is_best:
                        self.best_dev_bleu = dev_metrics.get('bleu', 0.0)
                        self.logger.info(f"  ✓ New best dev BLEU: {self.best_dev_bleu:.4f}")
                else:
                    is_best = dev_metrics['loss'] < self.best_dev_loss
                    if is_best:
                        self.best_dev_loss = dev_metrics['loss']
                        self.logger.info(f"  ✓ New best dev loss: {self.best_dev_loss:.4f}")
                
                self.save_checkpoint(is_best=is_best)
            else:
                self.save_checkpoint(is_best=False)
            
            # Update learning rate for epoch-based schedulers
            if self.scheduler and isinstance(self.scheduler, (optim.lr_scheduler.CosineAnnealingLR, 
                                                               optim.lr_scheduler.StepLR)):
                self.scheduler.step()
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("Training Complete!")
        if self.metric_for_best == 'bleu':
            self.logger.info(f"Best dev BLEU: {self.best_dev_bleu:.4f}")
        else:
            self.logger.info(f"Best dev loss: {self.best_dev_loss:.4f}")
        self.logger.info("=" * 80)