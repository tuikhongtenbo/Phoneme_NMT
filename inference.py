"""
Interactive inference script for NMT model.
Input an English sentence -> Output Vietnamese translation.

Usage:
    python inference.py --checkpoint checkpoints/best_model.pt --config configs/transformer_phoneme.yaml

Commands:
    :quit  or :q  - Exit
    :bleu        - Show last translation BLEU score
"""

import argparse
import torch
from pathlib import Path
import sys

from configs.config import Config
from src.models import TransformerModel, LSTMBahdanau, LSTMLuong, LSTMSeq2Seq
from src.training.trainer import Trainer
from src.utils.logger import setup_logger
from src.data.helpers import get_vocab_filepath
from src.data.bpe.vocab import BPEVocab
from src.data.phoneme.vocab import EnPhonemeVocab, ViWordVocab
from src.data.word.vocab import EnWordVocab, ViWordLevelVocab
from src.data.unigram.vocab import UnigramVocab
from src.data.helpers import create_vi_vocab_config
from src.data.text_utils import preprocess_sentence


# ─── Inference Engine ───────────────────────────────────────────────────────────

class NMTInferrer:
    """Load model + vocab and run greedy autoregressive decoding."""

    def __init__(self, checkpoint_path: str, config: Config,
                 input_vocab, output_vocab, target_level: str, logger):
        self.config = config
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.target_level = target_level
        self.logger = logger
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # Build model
        src_vocab_size = input_vocab.vocab_size if hasattr(input_vocab, 'vocab_size') else input_vocab.count
        tgt_vocab_size = output_vocab.vocab_size if hasattr(output_vocab, 'vocab_size') else output_vocab.count

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
            self.model = TransformerModel(
                config=model_config,
                src_vocab_size=src_vocab_size,
                tgt_vocab_size=tgt_vocab_size
            )
        elif model_name == "lstm_bahdanau":
            self.model = LSTMBahdanau(config=model_config,
                                      src_vocab_size=src_vocab_size,
                                      tgt_vocab_size=tgt_vocab_size)
        elif model_name == "lstm_luong":
            self.model = LSTMLuong(config=model_config,
                                   src_vocab_size=src_vocab_size,
                                   tgt_vocab_size=tgt_vocab_size)
        elif model_name == "lstm_seq2seq":
            self.model = LSTMSeq2Seq(config=model_config,
                                    src_vocab_size=src_vocab_size,
                                    tgt_vocab_size=tgt_vocab_size)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        self.model.to(self.device)
        self.model.eval()

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f"Loaded checkpoint: {checkpoint_path} "
                        f"(epoch {checkpoint.get('epoch','?')}, step {checkpoint.get('global_step','?')})")

        # Special tokens
        self.pad_id = config.data.pad_id
        self.sos_id = config.data.sos_id
        self.eos_id = config.data.eos_id
        self.max_len = config.data.max_seq_len

    def _encode_input(self, text: str) -> torch.Tensor:
        """Encode source sentence to tensor."""
        if self.config.data.source_level in ['word', 'bpe', 'unigram']:
            indices = self.input_vocab.sentence_to_indices(text)
            indices = [self.input_vocab.bos_idx] + indices + [self.input_vocab.eos_idx]
        else:  # phoneme
            indices = self.input_vocab.encode_caption(text)

        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        return torch.tensor([indices], dtype=torch.long, device=self.device)

    def _decode_indices(self, indices: list) -> str:
        """Decode index list to text string."""
        # Remove special tokens
        indices = [i for i in indices if i not in {self.pad_id, self.sos_id}]
        if self.eos_id in indices:
            indices = indices[:indices.index(self.eos_id)]

        # Handle phoneme vocab with native decode
        if hasattr(self.output_vocab, 'decode_caption'):
            try:
                return self.output_vocab.decode_caption(
                    torch.tensor(indices, dtype=torch.long), join_words=True
                )
            except Exception:
                pass

        # Handle pretrained tokenizer
        if hasattr(self.output_vocab, 'tokenizer'):
            try:
                return self.output_vocab.tokenizer.decode(indices, skip_special_tokens=True)
            except Exception:
                pass

        # Handle regular vocab
        special = {0, 1, 2, 3}
        if hasattr(self.output_vocab, 'index2word'):
            tokens = [self.output_vocab.index2word.get(i, '<UNK>') for i in indices if i not in special]
        elif hasattr(self.output_vocab, 'itos'):
            tokens = [self.output_vocab.itos.get(i, '<UNK>') for i in indices if i not in special]
        else:
            tokens = [str(i) for i in indices if i not in special]

        return ' '.join(tokens)

    def _greedy_decode(self, src_tensor: torch.Tensor) -> list:
        """Greedy autoregressive decoding: argmax at each step."""
        device = src_tensor.device
        model_name = self.config.model.name.lower()

        if model_name == "transformer":
            # Transformer: encode once, then autoregressive decode
            _ = self.model.encode(src_tensor)
            src_mask = (src_tensor != self.pad_id).unsqueeze(1)
            generated = [self.sos_id]
            for _ in range(self.max_len - 1):
                tgt_tensor = torch.tensor([generated], dtype=torch.long, device=device)
                logits = self.model(tgt_tensor, src_mask=src_mask)
                next_token = logits[0, -1, :].argmax(dim=-1).item()
                if next_token == self.eos_id:
                    break
                generated.append(next_token)

        elif model_name in ("lstm_seq2seq", "lstm_bahdanau", "lstm_luong"):
            # LSTM: encode source to get hidden/cell state
            # encode() returns (outputs, (hidden, cell)) — must unpack properly
            _, (hidden, cell) = self.model.encode(src_tensor)

            generated = [self.sos_id]
            decoder_input = torch.tensor([[self.sos_id]], dtype=torch.long, device=device)

            for _ in range(self.max_len - 1):
                # decode_step returns (prediction, (hidden, cell))
                prediction, (hidden, cell) = self.model.decode_step(
                    decoder_input, (_, (hidden, cell)), past_key_values=(hidden, cell)
                )
                # prediction shape: (1, vocab) or (batch, vocab)
                if prediction.dim() == 2 and prediction.size(0) > 1:
                    prediction = prediction.squeeze(0)
                next_token = prediction.argmax(dim=-1).item()
                if next_token == self.eos_id:
                    break
                generated.append(next_token)
                decoder_input = torch.tensor([[next_token]], dtype=torch.long, device=device)

        else:
            raise ValueError(f"Unsupported model for inference: {model_name}")

        return generated

    def translate(self, src_text: str) -> str:
        """Translate one English sentence to Vietnamese."""
        src_tensor = self._encode_input(src_text)
        pred_indices = self._greedy_decode(src_tensor)
        pred_text = self._decode_indices(pred_indices)
        return pred_text


# ─── Load vocabularies ─────────────────────────────────────────────────────────

def load_vocabularies(config: Config):
    source_level = getattr(config.data, 'source_level', 'word').lower()
    target_level = getattr(config.data, 'target_level', 'word').lower()
    min_count = getattr(config.data, 'min_count', 3)

    inp_path, out_path = get_vocab_filepath(source_level, target_level, min_count)

    def load_input():
        if source_level == 'word':
            return EnWordVocab.load(inp_path, name="en")
        elif source_level == 'phoneme':
            return EnPhonemeVocab.load(inp_path, config)
        elif source_level == 'bpe':
            return BPEVocab.load(inp_path, name="en_bpe")
        elif source_level == 'unigram':
            return UnigramVocab.load(inp_path, name="en_unigram")
        raise ValueError(f"Unknown source_level: {source_level}")

    def load_output():
        if target_level == 'word':
            return ViWordLevelVocab.load(out_path, name='vi_word')
        elif target_level == 'phoneme':
            return ViWordVocab.load(out_path, create_vi_vocab_config(config))
        elif target_level == 'bpe':
            return BPEVocab.load(out_path, name='vi_bpe')
        elif target_level == 'unigram':
            return UnigramVocab.load(out_path, name='vi_unigram')
        raise ValueError(f"Unknown target_level: {target_level}")

    return load_input(), load_output()


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NMT Inference — EN -> VI")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, default="configs/transformer_phoneme.yaml",
                        help="Path to config YAML")
    parser.add_argument("--max_len", type=int, default=None,
                        help="Override max generation length")
    parser.add_argument("--test_src",  type=str, default=None, help="Test source path (overrides config)")
    parser.add_argument("--test_tgt",  type=str, default=None, help="Test target path (overrides config)")
    parser.add_argument("--text", type=str, default=None,
                        help="English sentence to translate (non-interactive mode)")
    parser.add_argument("--file", type=str, default=None,
                        help="Text file with one sentence per line to translate (non-interactive mode)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for translations (used with --file)")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load config
    config = Config.from_yaml(args.config)
    if args.test_src is not None:
        config.data.test_src = args.test_src
    if args.test_tgt is not None:
        config.data.test_tgt = args.test_tgt
    if args.max_len is not None:
        config.data.max_seq_len = args.max_len

    # Setup logger
    logger = setup_logger(name="Inference")

    # Load vocabularies
    logger.info("Loading vocabularies...")
    input_vocab, output_vocab = load_vocabularies(config)
    target_level = getattr(config.data, 'target_level', 'word').lower()
    logger.info("Vocabularies loaded successfully.")

    # Build inferrer
    inferrer = NMTInferrer(
        checkpoint_path=str(checkpoint_path),
        config=config,
        input_vocab=input_vocab,
        output_vocab=output_vocab,
        target_level=target_level,
        logger=logger
    )

    device = inferrer.device
    sos_id = inferrer.sos_id
    eos_id = inferrer.eos_id
    pad_id = inferrer.pad_id

    print("\n" + "=" * 60)
    print("  NMT Inference — English to Vietnamese")
    print("=" * 60)
    print(f"  Model:      {config.model.name}")
    print(f"  Device:     {device}")
    print(f"  Checkpoint: {checkpoint_path}")
    print("=" * 60)
    print("  Commands: :quit  Exit")
    print("  Target level:", target_level.upper())
    print("=" * 60 + "\n")

    # ── Non-interactive: translate from --file ─────────────────────────────────
    if args.file:
        if not Path(args.file).exists():
            print(f"[ERROR] File not found: {args.file}")
            return
        output_file = args.output or args.file.replace(".txt", "_translated.txt")
        results = []
        with open(args.file, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]
        for i, line in enumerate(lines):
            print(f"[{i+1}/{len(lines)}] {line}")
            try:
                pred = inferrer.translate(line)
            except Exception as e:
                pred = f"[ERROR] {e}"
            print(f"  -> {pred}")
            results.append(pred)
        with open(output_file, 'w', encoding='utf-8') as f:
            for src, tgt in zip(lines, results):
                f.write(f"{src}\t{tgt}\n")
        print(f"\nSaved {len(results)} translations to: {output_file}")
        return

    # ── Non-interactive: translate single sentence via --text ───────────────────
    if args.text:
        print(f"[EN] > {args.text}")
        try:
            pred = inferrer.translate(args.text)
        except Exception as e:
            pred = f"[ERROR] {e}"
        print(f"[VI] > {pred}")
        return

    # ── Interactive loop ───────────────────────────────────────────────────────
    while True:
        try:
            user_input = input("\n[EN] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nExiting.")
            break

        if not user_input:
            continue

        if user_input.lower() in (':q', ':quit', ':exit'):
            print("Goodbye!")
            break

        # Encode source
        try:
            if config.data.source_level in ['word', 'bpe', 'unigram']:
                en_indices = input_vocab.sentence_to_indices(user_input)
                en_indices = [input_vocab.bos_idx] + en_indices + [input_vocab.eos_idx]
            else:
                en_indices = input_vocab.encode_caption(user_input)
                if isinstance(en_indices, torch.Tensor):
                    en_indices = en_indices.tolist()

            src_tensor = torch.tensor([en_indices], dtype=torch.long, device=device)
        except Exception as e:
            print(f"[ERROR] Encoding failed: {e}")
            continue

        # Decode
        try:
            pred_indices = inferrer._greedy_decode(src_tensor)
            pred_text = inferrer._decode_indices(pred_indices)
        except Exception as e:
            print(f"[ERROR] Decoding failed: {e}")
            continue

        print(f"[VI] > {pred_text}")


if __name__ == "__main__":
    main()
