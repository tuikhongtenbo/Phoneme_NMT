"""
Evaluation script for NMT models.
Translates test set and computes BLEU, METEOR, ROUGE-L scores.
"""

import argparse
import json
import torch
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

from configs.config import Config
from src.data.data_loader import prepare_data, create_data_loader
from src.models import TransformerModel, LSTMBahdanau, LSTMLuong, LSTMSeq2Seq
from src.evaluation.evaluator import Evaluator
from src.utils.logger import setup_logger


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(config: Config, checkpoint_path: str, src_vocab_size: int, tgt_vocab_size: int):
    """Load model from checkpoint."""
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

    model_name = config.model.name.lower()
    if model_name == "transformer":
        model = TransformerModel(config=model_config, src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size)
    elif model_name == "lstm_bahdanau":
        model = LSTMBahdanau(config=model_config, src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size)
    elif model_name == "lstm_luong":
        model = LSTMLuong(config=model_config, src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size)
    elif model_name == "lstm_seq2seq":
        model = LSTMSeq2Seq(config=model_config, src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def decode_indices(indices: list, vocab) -> str:
    """Decode a list of token IDs back to text."""
    special_tokens = {0, 1, 2, 3}  # PAD, SOS, EOS, UNK

    if hasattr(vocab, 'decode_caption'):
        try:
            return vocab.decode_caption(torch.tensor(indices, dtype=torch.long), join_words=True)
        except Exception:
            pass

    if hasattr(vocab, 'tokenizer'):
        try:
            decoded = vocab.tokenizer.decode(indices, skip_special_tokens=True)
            return decoded
        except Exception:
            pass

    if hasattr(vocab, 'index2word'):
        tokens = [vocab.index2word.get(idx, '<UNK>') for idx in indices if idx not in special_tokens]
    elif hasattr(vocab, 'itos'):
        tokens = [vocab.itos.get(idx, '<UNK>') for idx in indices if idx not in special_tokens]
    else:
        tokens = [str(idx) for idx in indices if idx not in special_tokens]

    return ' '.join(tokens)


def greedy_decode(model, src_seq: torch.Tensor, sos_id: int, eos_id: int, pad_id: int, max_len: int = 100) -> list:
    """Greedy decoding for a single source sequence."""
    device = next(model.parameters()).device
    src_seq = src_seq.to(device)

    with torch.no_grad():
        encoder_output = model.encode(src_seq)

        if isinstance(encoder_output, tuple):
            encoder_output, _ = encoder_output

        output_tokens = [sos_id]
        past_key_values = None

        for _ in range(max_len - 1):
            tgt_token = torch.tensor([[output_tokens[-1]]], device=device)
            logits, past_key_values = model.decode_step(
                tgt_token.squeeze(1),
                encoder_output,
                past_key_values
            )

            if isinstance(logits, torch.Tensor):
                next_token = logits.argmax(dim=-1).item()
            else:
                next_token = torch.argmax(logits, dim=-1).item()

            output_tokens.append(next_token)

            if next_token == eos_id or next_token == pad_id:
                break

    return output_tokens[1:]  # Remove SOS


def beam_decode(model, src_seq: torch.Tensor, sos_id: int, eos_id: int, pad_id: int,
                max_len: int = 100, beam_size: int = 5) -> str:
    """Beam search decoding for a single source sequence."""
    device = next(model.parameters()).device
    src_seq = src_seq.to(device)

    with torch.no_grad():
        encoder_output = model.encode(src_seq)

        if isinstance(encoder_output, tuple):
            encoder_output, _ = encoder_output

        # beams: list of (log_prob, token_sequence)
        beams = [(0.0, [sos_id])]
        completed = []

        for _ in range(max_len - 1):
            all_candidates = []

            for log_prob, seq in beams:
                if seq[-1] == eos_id:
                    completed.append((log_prob, seq))
                    continue

                tgt_token = torch.tensor([seq[-1]], device=device)
                logits, _ = model.decode_step(
                    tgt_token,
                    encoder_output,
                    None
                )

                if isinstance(logits, torch.Tensor):
                    log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
                    topk = log_probs.topk(beam_size)
                else:
                    log_probs = torch.log_softmax(torch.tensor(logits), dim=-1).squeeze(0)
                    topk = log_probs.topk(beam_size)

                for log_p, idx in zip(topk.values, topk.indices):
                    idx = idx.item()
                    new_seq = seq + [idx]
                    all_candidates.append((log_prob + log_p.item(), new_seq))

            if not all_candidates:
                break

            all_candidates.sort(key=lambda x: x[0], reverse=True)
            beams = all_candidates[:beam_size]

            if len(beams) == 0:
                break

        all_candidates = completed + beams
        if not all_candidates:
            return ""

        all_candidates.sort(key=lambda x: x[0] / (len(x[1]) ** 0.7 + 1e-9), reverse=True)
        best_seq = all_candidates[0][1]

        return [t for t in best_seq if t != sos_id and t != eos_id]


def main():
    parser = argparse.ArgumentParser(description="Evaluate NMT Model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--split", type=str, default="test", choices=["train", "dev", "test"],
                        help="Dataset split to evaluate")
    parser.add_argument("--beam_size", type=int, default=1, help="Beam size (1 = greedy)")
    parser.add_argument("--max_len", type=int, default=100, help="Max decode length")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size override")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    args = parser.parse_args()

    # Load config
    config = Config.from_yaml(args.config)

    # Overrides
    if args.seed is not None:
        config.seed = args.seed
    set_seed(config.seed or 42)

    if args.batch_size is not None:
        config.training.batch_size = args.batch_size

    # Setup output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_name = Path(args.config).stem
    exp_name = f"{config_name}_beam{args.beam_size}"
    predictions_path = output_dir / f"{exp_name}_predictions.txt"
    metrics_path = output_dir / f"{exp_name}_metrics.json"

    logger = setup_logger(output=str(output_dir), name=f"Eval_{exp_name}")

    logger.info("=" * 60)
    logger.info("NMT Evaluation")
    logger.info("=" * 60)
    logger.info(f"Config: {args.config}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Beam size: {args.beam_size}")
    logger.info("=" * 60)

    # Prepare data
    logger.info("Preparing data...")
    data_result = prepare_data(
        splits=[args.split],
        max_len=config.data.max_seq_len,
        min_count=config.data.min_count,
        config=config
    )

    input_vocab = data_result['input_vocab']
    output_vocab = data_result['output_vocab']
    indexed_data = data_result['data']
    target_level = data_result['target_level']

    src_vocab_size = getattr(input_vocab, 'vocab_size', getattr(input_vocab, 'count', 0))
    tgt_vocab_size = getattr(output_vocab, 'vocab_size', getattr(output_vocab, 'count', 0))

    logger.info(f"Source vocab size: {src_vocab_size}")
    logger.info(f"Target vocab size: {tgt_vocab_size}")
    logger.info(f"Target level: {target_level}")
    logger.info(f"Eval samples: {len(indexed_data[args.split])}")

    # Load model
    logger.info("Loading model...")
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = load_model(config, args.checkpoint, src_vocab_size, tgt_vocab_size)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # Create data loader for batching
    data_loader = create_data_loader(
        indexed_pairs=indexed_data[args.split],
        batch_size=config.training.batch_size,
        shuffle=False,
        target_level=target_level
    )

    sos_id = config.data.sos_id
    eos_id = config.data.eos_id
    pad_id = config.data.pad_id

    # Translate
    logger.info(f"Translating ({'beam' if args.beam_size > 1 else 'greedy'} decoding)...")
    all_predictions = []
    all_references = []

    for batch in tqdm(data_loader, desc="Translating", dynamic_ncols=True):
        src_seq, tgt_seq = batch
        batch_size = src_seq.size(0)

        for i in range(batch_size):
            src_i = src_seq[i:i+1]
            tgt_i = tgt_seq[i]

            # Get reference (remove padding)
            ref_ids = tgt_i[tgt_i != pad_id].cpu().tolist()
            if eos_id in ref_ids:
                ref_ids = ref_ids[:ref_ids.index(eos_id)]
            reference = decode_indices(ref_ids, output_vocab)

            # Decode
            if args.beam_size > 1:
                pred_ids = beam_decode(model, src_i, sos_id, eos_id, pad_id, args.max_len, args.beam_size)
            else:
                pred_ids = greedy_decode(model, src_i, sos_id, eos_id, pad_id, args.max_len)

            prediction = decode_indices(pred_ids, output_vocab)

            all_predictions.append(prediction)
            all_references.append(reference)

    # Save predictions
    with open(predictions_path, 'w', encoding='utf-8') as f:
        for pred in all_predictions:
            f.write(pred + '\n')
    logger.info(f"Saved predictions to: {predictions_path}")

    # Compute metrics
    logger.info("Computing metrics...")
    evaluator = Evaluator(metrics=['bleu', 'rouge_l', 'meteor'])

    try:
        scores = evaluator.evaluate(all_references, all_predictions)
    except Exception as e:
        logger.warning(f"Evaluator error: {e}")
        scores = {}

    # BLEU scores
    try:
        from sacrebleu import corpus_bleu
        bleu = corpus_bleu(all_predictions, [all_references])
        scores['bleu_1'] = bleu.subscores['bp'] * bleu.precisions[0] / 100 if len(bleu.precisions) > 0 else 0
        scores['bleu_2'] = bleu.subscores['bp'] * bleu.precisions[1] / 100 if len(bleu.precisions) > 1 else 0
        scores['bleu_3'] = bleu.subscores['bp'] * bleu.precisions[2] / 100 if len(bleu.precisions) > 2 else 0
        scores['bleu_4'] = bleu.subscores['bp'] * bleu.precisions[3] / 100 if len(bleu.precisions) > 3 else 0
        scores['bleu'] = bleu.score / 100  # sacrebleu BLEU is 0-100 scale
    except ImportError:
        logger.warning("sacrebleu not installed, skipping BLEU")
    except Exception as e:
        logger.warning(f"BLEU computation error: {e}")

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    for key, value in sorted(scores.items()):
        logger.info(f"  {key.upper()}: {value:.4f}")

    # Save metrics
    metrics_output = {
        'config': str(args.config),
        'checkpoint': str(args.checkpoint),
        'split': args.split,
        'beam_size': args.beam_size,
        'num_samples': len(all_predictions),
        'metrics': scores
    }
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_output, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSaved metrics to: {metrics_path}")

    # Show sample predictions
    logger.info("\n--- Sample Predictions ---")
    for i in range(min(5, len(all_predictions))):
        logger.info(f"[{i}] REF: {all_references[i]}")
        logger.info(f"[{i}] PRED: {all_predictions[i]}")
        logger.info("")


if __name__ == "__main__":
    main()
