"""
Main entry point for training Neural Machine Translation models.
Supports Transformer, LSTM-Bahdanau, and LSTM-Luong architectures.
"""

import argparse
import torch
import random
import numpy as np
from pathlib import Path

from configs.config import Config
from src.data.data_loader import prepare_data, create_data_loader
from src.models import TransformerModel, LSTMBahdanau, LSTMLuong, LSTMSeq2Seq
from src.training.trainer import Trainer
from src.utils.logger import setup_logger


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(config: Config, src_vocab_size: int, tgt_vocab_size: int):
    """
    Create model based on configuration.
    
    Args:
        config: Configuration object
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        
    Returns:
        Model instance
    """
    model_name = config.model.name.lower()
    
    # Convert config to dict format expected by models
    model_config = {
        "model.embed_dim": config.model.embed_dim,
        "model.hidden_dim": config.model.hidden_dim,
        "model.num_layers": config.model.num_layers,
        "model.dropout": config.model.dropout,
        "model.attention_type": config.model.attention_type,
        "model.num_heads": config.model.num_heads,
        "model.ff_dim": config.model.ff_dim,
    }
    
    # Add encoder_layers and decoder_layers if they exist in config
    if hasattr(config.model, 'encoder_layers'):
        model_config["model.encoder_layers"] = config.model.encoder_layers
    if hasattr(config.model, 'decoder_layers'):
        model_config["model.decoder_layers"] = config.model.decoder_layers
    
    if model_name == "transformer":
        model_config["data.max_seq_len"] = config.data.max_seq_len
        model = TransformerModel(
            config=model_config,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size
        )
    elif model_name == "lstm_bahdanau" or (model_name == "lstm" and config.model.attention_type == "bahdanau"):
        model = LSTMBahdanau(
            config=model_config,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size
        )
    elif model_name == "lstm_luong" or (model_name == "lstm" and config.model.attention_type in ["general", "dot", "concat"]):
        model = LSTMLuong(
            config=model_config,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size
        )
    elif model_name == "lstm_seq2seq" or (model_name == "lstm" and config.model.attention_type in ["none", "None", None, ""]):
        model = LSTMSeq2Seq(
            config=model_config,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}. "
                        f"Supported: 'transformer', 'lstm_bahdanau', 'lstm_luong', 'lstm_seq2seq'")
    
    return model


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train NMT Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/transformer.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (uses config seed if not provided)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config)"
    )
    parser.add_argument(
        "--src_level",
        type=str,
        choices=["word", "phoneme", "bpe", "unigram"],
        default=None,
        help="Source sequence level (overrides config)"
    )
    parser.add_argument(
        "--tgt_level",
        type=str,
        choices=["word", "phoneme", "bpe", "unigram"],
        default=None,
        help="Target sequence level (overrides config)"
    )
    parser.add_argument(
        "--pretrained_mode",
        type=str,
        choices=["pretrained_1", "pretrained_2"],
        default=None,
        help="'pretrained_1' (mBART->mBART) or 'pretrained_2' (mBART->BARTPho)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help="Evaluate every N steps (overrides config.training.eval_every)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Maximum sequence length (overrides config.data.max_seq_len)"
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=None,
        help="Maximum sequence length alias for --max_length (overrides config.data.max_seq_len)"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help="Save checkpoint every N steps (overrides config.training.save_every)"
    )
    parser.add_argument(
        "--train_src",
        type=str,
        default=None,
        help="Path to training source file (overrides config.data.train_src)"
    )
    parser.add_argument(
        "--train_tgt",
        type=str,
        default=None,
        help="Path to training target file (overrides config.data.train_tgt)"
    )
    parser.add_argument(
        "--dev_src",
        type=str,
        default=None,
        help="Path to dev source file (overrides config.data.dev_src)"
    )
    parser.add_argument(
        "--dev_tgt",
        type=str,
        default=None,
        help="Path to dev target file (overrides config.data.dev_tgt)"
    )
    parser.add_argument(
        "--test_src",
        type=str,
        default=None,
        help="Path to test source file (overrides config.data.test_src)"
    )
    parser.add_argument(
        "--test_tgt",
        type=str,
        default=None,
        help="Path to test target file (overrides config.data.test_tgt)"
    )

    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = Config.from_yaml(config_path)
    
    # Parsing with command line arguments
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.num_epochs is not None:
        config.training.num_epochs = args.num_epochs
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.eval_steps is not None:
        config.training.eval_every = args.eval_steps
    if args.max_length is not None:
        config.data.max_seq_len = args.max_length
    if args.max_seq_len is not None:
        config.data.max_seq_len = args.max_seq_len
    if args.save_steps is not None:
        config.training.save_every = args.save_steps
    if args.train_src is not None:
        config.data.train_src = args.train_src
    if args.train_tgt is not None:
        config.data.train_tgt = args.train_tgt
    if args.dev_src is not None:
        config.data.dev_src = args.dev_src
    if args.dev_tgt is not None:
        config.data.dev_tgt = args.dev_tgt
    if args.test_src is not None:
        config.data.test_src = args.test_src
    if args.test_tgt is not None:
        config.data.test_tgt = args.test_tgt
    if args.pretrained_mode is not None:
        config.data.tokenizer_type = args.pretrained_mode
    else:
        if args.src_level is not None:
            config.data.source_level = args.src_level
            config.data.tokenizer_type = None
        if args.tgt_level is not None:
            config.data.target_level = args.tgt_level
            config.data.tokenizer_type = None
    
    # Set seed
    seed = args.seed if args.seed is not None else (config.seed if config.seed else 42)
    set_seed(seed)
    
    # Setup logger
    log_dir = Path("logs") / config.model.name
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output=str(log_dir), name="PhonemeNMT")
    
    logger.info("=" * 80)
    logger.info("Phoneme NMT Training")
    logger.info("=" * 80)
    logger.info(f"Configuration: {config_path}")
    logger.info(f"Model: {config.model.name}")
    logger.info(f"Batch size: {config.training.batch_size}")
    logger.info(f"Number of epochs: {config.training.num_epochs}")
    if config.data.tokenizer_type:
        logger.info(f"Tokenizer type: {config.data.tokenizer_type}")
        logger.info(f"  - pretrained_1: mBART (EN) -> mBART (VI)")
        logger.info(f"  - pretrained_2: mBART (EN) -> BARTPho (VI)")
    else:
        logger.info(f"Source Level: {config.data.source_level}")
        logger.info(f"Target Level: {config.data.target_level}")
    logger.info(f"Seed: {seed}")
    logger.info("=" * 80)
    
    # Prepare data
    logger.info("\nPreparing data...")
    data_splits = ['train', 'dev']
    if hasattr(config.data, 'test_src') and config.data.test_src:
        data_splits.append('test')
    
    data_result = prepare_data(
        splits=data_splits,
        max_len=config.data.max_seq_len,
        min_count=config.data.min_count,
        config=config,
        limit_train=getattr(config.data, 'limit_train', None) or None
    )
    
    input_vocab = data_result['input_vocab']
    output_vocab = data_result['output_vocab']
    indexed_data = data_result['data']
    target_level = data_result['target_level']
    src_vocab_size = input_vocab.vocab_size if hasattr(input_vocab, 'vocab_size') else input_vocab.count
    tgt_vocab_size = output_vocab.vocab_size if hasattr(output_vocab, 'vocab_size') else output_vocab.count
    
    logger.info(f"Source vocabulary size: {src_vocab_size}")
    logger.info(f"Target vocabulary size: {tgt_vocab_size}")
    logger.info(f"Target level: {target_level}")
    logger.info(f"Train pairs: {len(indexed_data['train'])}")
    logger.info(f"Dev pairs: {len(indexed_data['dev'])}")
    if 'test' in indexed_data:
        logger.info(f"Test pairs: {len(indexed_data['test'])}")
    
    # Create data loaders
    logger.info("\nCreating data loaders...")
    train_loader = create_data_loader(
        indexed_pairs=indexed_data['train'],
        batch_size=config.training.batch_size,
        shuffle=True,
        target_level=target_level
    )
    
    dev_loader = create_data_loader(
        indexed_pairs=indexed_data['dev'],
        batch_size=config.training.batch_size,
        shuffle=False,
        target_level=target_level
    )
    
    test_loader = None
    if 'test' in indexed_data:
        test_loader = create_data_loader(
            indexed_pairs=indexed_data['test'],
            batch_size=config.training.batch_size,
            shuffle=False,
            target_level=target_level
        )
    
    # Create model
    logger.info("\nCreating model...")
    model = create_model(config, src_vocab_size, tgt_vocab_size)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Update config with actual target_level from preprocessing 
    if target_level != config.data.target_level:
        logger.info(f"Note: Actual target_level ({target_level}) differs from config ({config.data.target_level}). "
                   f"Using actual level from preprocessing.")
        config.data.target_level = target_level
    
    # Create trainer
    logger.info("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        dev_loader=dev_loader,
        logger=logger,
        input_vocab=input_vocab,
        output_vocab=output_vocab,
        target_level=target_level  # Pass actual target_level from preprocessing
    )
    
    # Start training
    logger.info("\nStarting training...")
    trainer.train(resume_from=args.resume)

    logger.info("\nTraining completed successfully!")
    logger.info(f"Best model saved to: {trainer.checkpoint_dir / 'best_model.pt'}")

    # ── Evaluate on test set ────────────────────────────────────────────────
    if test_loader is not None:
        best_model_path = trainer.checkpoint_dir / "best_model.pt"
        if best_model_path.exists():
            logger.info("\n" + "=" * 80)
            logger.info("Loading best model for test evaluation...")
            logger.info("=" * 80)

            # Load best checkpoint (only model state dict needed)
            checkpoint = torch.load(best_model_path, map_location='cpu')
            trainer.model.load_state_dict(checkpoint['model_state_dict'])

            # Evaluate on test set
            logger.info("Evaluating on test set...")
            test_metrics = trainer.evaluate(test_loader)

            logger.info("\n" + "=" * 80)
            logger.info("TEST SET RESULTS")
            logger.info("=" * 80)
            logger.info(f"  Test Loss:       {test_metrics['loss']:.4f}")
            logger.info(f"  Test Perplexity: {test_metrics['perplexity']:.2f}")
            logger.info(f"  Test BLEU:       {test_metrics.get('bleu', 0.0):.4f}")
            if 'bleu_1' in test_metrics:
                logger.info(f"  BLEU-1: {test_metrics['bleu_1']:.4f}")
                logger.info(f"  BLEU-2: {test_metrics['bleu_2']:.4f}")
                logger.info(f"  BLEU-3: {test_metrics['bleu_3']:.4f}")
                logger.info(f"  BLEU-4: {test_metrics['bleu_4']:.4f}")
            if 'rouge_l' in test_metrics:
                logger.info(f"  ROUGE-L: {test_metrics['rouge_l']:.4f}")
            if 'meteor' in test_metrics:
                logger.info(f"  METEOR:  {test_metrics['meteor']:.4f}")
            logger.info("=" * 80)

            # Save results
            results_path = trainer.checkpoint_dir / "test_results.txt"
            with open(results_path, "w", encoding="utf-8") as f:
                f.write(f"Checkpoint: {best_model_path}\n")
                f.write(f"Test Loss: {test_metrics['loss']:.4f}\n")
                f.write(f"Test Perplexity: {test_metrics['perplexity']:.2f}\n")
                f.write(f"Test BLEU: {test_metrics.get('bleu', 0.0):.4f}\n")
                if 'bleu_1' in test_metrics:
                    f.write(f"BLEU-1: {test_metrics['bleu_1']:.4f}\n")
                    f.write(f"BLEU-2: {test_metrics['bleu_2']:.4f}\n")
                    f.write(f"BLEU-3: {test_metrics['bleu_3']:.4f}\n")
                    f.write(f"BLEU-4: {test_metrics['bleu_4']:.4f}\n")
                if 'rouge_l' in test_metrics:
                    f.write(f"ROUGE-L: {test_metrics['rouge_l']:.4f}\n")
                if 'meteor' in test_metrics:
                    f.write(f"METEOR: {test_metrics['meteor']:.4f}\n")
            logger.info(f"Results saved to: {results_path}")
        else:
            logger.warning(
                f"best_model.pt not found at {best_model_path}. "
                "Test evaluation skipped. (This may happen if no checkpoint improved over the initial model.)"
            )
    else:
        logger.info("No test data available. Test evaluation skipped.")


if __name__ == "__main__":
    main()