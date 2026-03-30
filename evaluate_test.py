"""
Evaluate trained NMT model on test set.
Usage:
    python evaluate_test.py --checkpoint checkpoints/best_model.pt --config configs/transformer_phoneme.yaml
"""

import argparse
import torch
from pathlib import Path

from configs.config import Config
from src.data.data_loader import prepare_data, create_data_loader
from src.models import TransformerModel, LSTMBahdanau, LSTMLuong, LSTMSeq2Seq
from src.training.trainer import Trainer
from src.utils.logger import setup_logger


def create_model(config: Config, src_vocab_size: int, tgt_vocab_size: int):
    model_name = config.model.name.lower()
    model_config = {
        "model.embed_dim": config.model.embed_dim,
        "model.hidden_dim": config.model.hidden_dim,
        "model.num_layers": config.model.num_layers,
        "model.dropout": config.model.dropout,
        "model.attention_type": config.model.attention_type,
        "model.num_heads": config.model.num_heads,
        "model.ff_dim": config.model.ff_dim,
    }
    if hasattr(config.model, 'encoder_layers'):
        model_config["model.encoder_layers"] = config.model.encoder_layers
    if hasattr(config.model, 'decoder_layers'):
        model_config["model.decoder_layers"] = config.model.decoder_layers
    if model_name == "transformer":
        model_config["data.max_seq_len"] = config.data.max_seq_len
        return TransformerModel(
            config=model_config,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size
        )
    elif model_name == "lstm_bahdanau":
        return LSTMBahdanau(
            config=model_config,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size
        )
    elif model_name == "lstm_luong":
        return LSTMLuong(
            config=model_config,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size
        )
    elif model_name == "lstm_seq2seq":
        return LSTMSeq2Seq(
            config=model_config,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate NMT Model on Test Set")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt file)")
    parser.add_argument("--config", type=str, default="configs/transformer_phoneme.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size (overrides config)")
    parser.add_argument("--train_src", type=str, default=None, help="Path to train source (overrides config)")
    parser.add_argument("--train_tgt", type=str, default=None, help="Path to train target (overrides config)")
    parser.add_argument("--dev_src",   type=str, default=None, help="Path to dev source (overrides config)")
    parser.add_argument("--dev_tgt",   type=str, default=None, help="Path to dev target (overrides config)")
    parser.add_argument("--test_src",  type=str, default=None, help="Path to test source (overrides config)")
    parser.add_argument("--test_tgt",  type=str, default=None, help="Path to test target (overrides config)")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load config
    config = Config.from_yaml(args.config)

    # Override batch size if provided
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
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

    # Setup logger
    log_dir = Path("logs") / config.model.name
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output=str(log_dir), name="EvalTest")

    logger.info("=" * 80)
    logger.info("Evaluating on Test Set")
    logger.info("=" * 80)

    # Prepare data (train + dev + test)
    logger.info("\nPreparing data...")
    data_result = prepare_data(
        splits=['train', 'dev', 'test'],
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

    logger.info(f"Source vocab size: {src_vocab_size}")
    logger.info(f"Target vocab size: {tgt_vocab_size}")
    logger.info(f"Target level: {target_level}")
    logger.info(f"Test pairs: {len(indexed_data.get('test', []))}")

    if 'test' not in indexed_data or len(indexed_data['test']) == 0:
        logger.error("No test data found! Check config.data.test_src path.")
        return

    # Create test data loader
    test_loader = create_data_loader(
        indexed_pairs=indexed_data['test'],
        batch_size=config.training.batch_size,
        shuffle=False,
        target_level=target_level
    )

    # Create model
    logger.info("\nCreating model...")
    model = create_model(config, src_vocab_size, tgt_vocab_size)

    # Load checkpoint
    logger.info(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded model from epoch {checkpoint.get('epoch', '?')}, "
                f"step {checkpoint.get('global_step', '?')}")

    # Create trainer (for evaluation logic) and set model
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=test_loader,  # dummy, won't be used
        dev_loader=None,
        logger=logger,
        input_vocab=input_vocab,
        output_vocab=output_vocab,
        target_level=target_level
    )

    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)

    logger.info("\n" + "=" * 80)
    logger.info("TEST SET RESULTS")
    logger.info("=" * 80)
    logger.info(f"  Test Loss:      {test_metrics['loss']:.4f}")
    logger.info(f"  Test Perplexity: {test_metrics['perplexity']:.2f}")
    logger.info(f"  Test BLEU:      {test_metrics.get('bleu', 0.0):.4f}")
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
    results_path = checkpoint_path.parent / "test_results.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(f"Checkpoint: {checkpoint_path}\n")
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
    logger.info(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
