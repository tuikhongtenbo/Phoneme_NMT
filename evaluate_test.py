"""
Evaluate trained NMT model on test set.
Only loads the saved vocabulary and encodes the TEST split — no train/dev encoding.

Usage:
    python evaluate_test.py \
        --checkpoint checkpoints/best_model.pt \
        --config configs/transformer_phoneme.yaml \
        --test_src dataset/vocabs/clean/test_end_clean.en \
        --test_tgt dataset/vocabs/clean/test_end_clean.vi
"""

import argparse
import os
import sys
import torch
from pathlib import Path
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from configs.config import Config
from src.data.data_loader import create_data_loader
from src.models import TransformerModel, LSTMBahdanau, LSTMLuong, LSTMSeq2Seq
from src.training.trainer import Trainer
from src.utils.logger import setup_logger
from src.data.helpers import get_vocab_filepath
from src.data.constants import PAD_ID
from src.data.bpe.vocab import BPEVocab
from src.data.phoneme.vocab import EnPhonemeVocab, ViWordVocab
from src.data.word.vocab import EnWordVocab, ViWordLevelVocab
from src.data.unigram.vocab import UnigramVocab
from src.data.helpers import create_vi_vocab_config
from src.data.text_utils import preprocess_sentence


# ─── Minimal dataset for test only ────────────────────────────────────────────

class TestDataset(Dataset):
    def __init__(self, indexed_pairs):
        self.pairs = indexed_pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def collate_test(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=PAD_ID)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_ID)
    return src_padded, tgt_padded


def create_test_loader(indexed_pairs, batch_size):
    dataset = TestDataset(indexed_pairs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_test)


def load_test_pairs(test_src: str, test_tgt: str) -> List[Tuple[str, str]]:
    """Load raw test sentence pairs."""
    if not os.path.exists(test_src):
        raise FileNotFoundError(f"Test source not found: {test_src}")
    if not os.path.exists(test_tgt):
        raise FileNotFoundError(f"Test target not found: {test_tgt}")

    print(f"Loading test data:")
    print(f"  Source: {test_src}")
    print(f"  Target: {test_tgt}")

    with open(test_src, 'r', encoding='utf-8') as f:
        en_lines = [line.strip() for line in f if line.strip()]
    with open(test_tgt, 'r', encoding='utf-8') as f:
        vi_lines = [line.strip() for line in f if line.strip()]

    if len(en_lines) != len(vi_lines):
        raise ValueError(f"Sentence count mismatch: {len(en_lines)} vs {len(vi_lines)}")

    print(f"[OK] Loaded {len(en_lines)} test sentence pairs")
    return list(zip(en_lines, vi_lines))


def encode_test_pairs(
    pairs: List[Tuple[str, str]],
    input_vocab,
    output_vocab,
    source_level: str,
    target_level: str,
    max_len: int
) -> List[Tuple[List[int], List[int]]]:
    """Encode raw test pairs to index sequences."""
    indexed = []
    progress_bar = tqdm(pairs, desc="Encoding test", file=sys.stdout, mininterval=1.0, ncols=100)

    for en_sent, vi_sent in progress_bar:
        try:
            # Source encoding
            if source_level in ['word', 'bpe', 'unigram']:
                en_indices = input_vocab.sentence_to_indices(en_sent)
                en_indices = [input_vocab.bos_idx] + en_indices + [input_vocab.eos_idx]
            else:  # phoneme
                en_indices = input_vocab.encode_caption(en_sent)

            # Target encoding
            vi_words = preprocess_sentence(vi_sent)
            vi_indices = output_vocab.encode_caption(vi_words)
            if isinstance(vi_indices, torch.Tensor):
                vi_indices = vi_indices.tolist()

            if source_level in ['word', 'bpe', 'unigram']:
                en_len = len(en_indices)
            else:
                if isinstance(en_indices[0], list):
                    en_len = sum(len(x) if isinstance(x, list) else 1 for x in en_indices)
                else:
                    en_len = len(en_indices)

            if target_level in ['word', 'bpe', 'unigram', 'pretrained']:
                vi_len = len(vi_indices)
            else:  # phoneme
                if isinstance(vi_indices, torch.Tensor):
                    vi_len = vi_indices.size(0)
                elif isinstance(vi_indices, list):
                    vi_len = len(vi_indices)
                else:
                    vi_len = 0

            if en_len <= max_len and vi_len <= max_len:
                indexed.append((en_indices, vi_indices))
        except Exception:
            continue

    progress_bar.close()
    return indexed


def load_vocabularies(config: Config, source_level: str, target_level: str, min_count: int):
    """Load pre-saved vocabularies (do NOT rebuild from train data)."""
    input_vocab_path, output_vocab_path = get_vocab_filepath(source_level, target_level, min_count)

    print(f"\nLoading input vocabulary from: {input_vocab_path}")
    print(f"Loading output vocabulary from: {output_vocab_path}")

    if not os.path.exists(input_vocab_path):
        raise FileNotFoundError(f"Input vocab not found: {input_vocab_path}. "
                                "Please run training first to create the vocabulary.")
    if not os.path.exists(output_vocab_path):
        raise FileNotFoundError(f"Output vocab not found: {output_vocab_path}. "
                                "Please run training first to create the vocabulary.")

    # Load input vocab
    if source_level == 'word':
        input_vocab = EnWordVocab.load(input_vocab_path, name="en")
    elif source_level == 'phoneme':
        input_vocab = EnPhonemeVocab.load(input_vocab_path, config)
    elif source_level == 'bpe':
        input_vocab = BPEVocab.load(input_vocab_path, name="en_bpe")
    elif source_level == 'unigram':
        input_vocab = UnigramVocab.load(input_vocab_path, name="en_unigram")
    else:
        raise ValueError(f"Unknown source_level: {source_level}")

    # Load output vocab
    if target_level == 'word':
        output_vocab = ViWordLevelVocab.load(output_vocab_path, name='vi_word')
    elif target_level == 'phoneme':
        vi_vocab_config = create_vi_vocab_config(config)
        output_vocab = ViWordVocab.load(output_vocab_path, vi_vocab_config)
    elif target_level == 'bpe':
        output_vocab = BPEVocab.load(output_vocab_path, name='vi_bpe')
    elif target_level == 'unigram':
        output_vocab = UnigramVocab.load(output_vocab_path, name='vi_unigram')
    else:
        raise ValueError(f"Unknown target_level: {target_level}")

    src_vocab_size = input_vocab.vocab_size if hasattr(input_vocab, 'vocab_size') else input_vocab.count
    tgt_vocab_size = output_vocab.vocab_size if hasattr(output_vocab, 'vocab_size') else output_vocab.count

    print(f"[OK] Input vocab size:  {src_vocab_size}")
    print(f"[OK] Output vocab size: {tgt_vocab_size}")

    return input_vocab, output_vocab, src_vocab_size, tgt_vocab_size


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
    parser.add_argument("--test_src",  type=str, default=None, help="Path to test source (overrides config)")
    parser.add_argument("--test_tgt",  type=str, default=None, help="Path to test target (overrides config)")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load config
    config = Config.from_yaml(args.config)

    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.test_src is not None:
        config.data.test_src = args.test_src
    if args.test_tgt is not None:
        config.data.test_tgt = args.test_tgt

    # Determine tokenization levels
    source_level = getattr(config.data, 'source_level', 'word').lower()
    target_level = getattr(config.data, 'target_level', 'word').lower()
    min_count = getattr(config.data, 'min_count', 3)

    # Setup logger
    log_dir = Path("logs") / config.model.name
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output=str(log_dir), name="EvalTest")

    logger.info("=" * 80)
    logger.info("Evaluating on Test Set")
    logger.info("=" * 80)
    logger.info(f"Test source: {config.data.test_src}")
    logger.info(f"Test target: {config.data.test_tgt}")

    # ── Step 1: Load saved vocabularies (no train/dev needed) ─────────────────
    logger.info("\nLoading saved vocabularies...")
    input_vocab, output_vocab, src_vocab_size, tgt_vocab_size = load_vocabularies(
        config, source_level, target_level, min_count
    )

    # ── Step 2: Load & encode test pairs only ─────────────────────────────────
    logger.info("\nLoading and encoding test data...")
    test_pairs = load_test_pairs(config.data.test_src, config.data.test_tgt)
    indexed_test = encode_test_pairs(
        pairs=test_pairs,
        input_vocab=input_vocab,
        output_vocab=output_vocab,
        source_level=source_level,
        target_level=target_level,
        max_len=config.data.max_seq_len
    )
    logger.info(f"Encoded {len(indexed_test)} test pairs")

    if not indexed_test:
        logger.error("No valid test pairs after encoding. Check max_seq_len.")
        return

    # ── Step 3: Create test data loader ───────────────────────────────────────
    test_loader = create_test_loader(indexed_test, config.training.batch_size)

    # ── Step 4: Create model ──────────────────────────────────────────────────
    logger.info("\nCreating model...")
    model = create_model(config, src_vocab_size, tgt_vocab_size)

    # ── Step 5: Load checkpoint ───────────────────────────────────────────────
    logger.info(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded model from epoch {checkpoint.get('epoch', '?')}, "
                f"step {checkpoint.get('global_step', '?')}")

    # ── Step 6: Evaluate ─────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=test_loader,  # dummy, not used
        dev_loader=None,
        logger=logger,
        input_vocab=input_vocab,
        output_vocab=output_vocab,
        target_level=target_level
    )

    logger.info("\nEvaluating on test set...")
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
