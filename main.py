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
from src.data.preprocessing import prepare_data
from src.data.data_loader import create_data_loader
from src.models import TransformerModel, LSTMBahdanau, LSTMLuong
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
    
    if model_name == "transformer":
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
    else:
        raise ValueError(f"Unknown model name: {model_name}. "
                        f"Supported: 'transformer', 'lstm_bahdanau', 'lstm_luong'")
    
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
        "--level",
        type=str,
        choices=["word", "phoneme"],
        default=None,
        help="Sequence level: 'word' or 'phoneme' (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = Config.from_yaml(config_path)
    
    # Override config with command line arguments
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.num_epochs is not None:
        config.training.num_epochs = args.num_epochs
    if args.level is not None:
        config.data.source_level = args.level
        config.data.target_level = args.level
    
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
    logger.info(f"Level: {config.data.source_level} (source and target)")
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
        config=config
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
    
    # Create trainer
    logger.info("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        dev_loader=dev_loader,
        logger=logger,
        input_vocab=input_vocab,
        output_vocab=output_vocab
    )
    
    # Start training
    logger.info("\nStarting training...")
    trainer.train(resume_from=args.resume)
    
    logger.info("\nTraining completed successfully!")
    logger.info(f"Best model saved to: {trainer.checkpoint_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()