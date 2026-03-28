# English-Vietnamese Neural Machine Translation

A research project implementing baseline NMT models for English-Vietnamese translation, supporting **BPE**, **Unigram**, and **Phoneme** tokenization.

## Overview

- **LSTM + Bahdanau Attention**: Seq2seq with additive attention
- **LSTM + Luong Attention**: Seq2seq with multiplicative attention (general / dot / concat)
- **Transformer**: Attention-based architecture (Vaswani et al., 2017)

All models support BPE, Unigram, and Phoneme tokenization, with BLEU / ROUGE / METEOR evaluation.

## Project Structure

```
Phoneme_NMT/
├── src/
│   ├── models/
│   │   ├── base_model.py           # Abstract base class
│   │   ├── attention/              # Bahdanau & Luong attention
│   │   ├── lstm/                   # LSTM Seq2Seq, Bahdanau, Luong
│   │   └── transformer/            # Transformer (encoder/decoder/layers/embedding)
│   ├── data/
│   │   ├── data_loader.py         # Main data loading utilities
│   │   ├── base_vocab.py           # Base vocabulary class
│   │   ├── text_utils.py           # Sentence preprocessing
│   │   ├── constants.py            # Special tokens & IDs
│   │   ├── helpers.py              # Config helpers
│   │   ├── word/vocab.py           # Word-level vocab (En + Vi)
│   │   ├── bpe/vocab.py            # BPE vocab (HuggingFace tokenizers)
│   │   ├── unigram/vocab.py        # Unigram vocab (HuggingFace tokenizers)
│   │   └── phoneme/
│   │       ├── en_vocab.py         # English phoneme vocab
│   │       ├── vi_vocab.py         # Vietnamese phoneme vocab
│   │       ├── english_utils.py    # English IPA -> phoneme
│   │       └── vietnamese_utils.py # Vietnamese syllable analyzer
│   ├── training/trainer.py         # Training loop
│   ├── evaluation/                 # BLEU, ROUGE, METEOR
│   └── utils/logger.py             # Logging
├── configs/                         # YAML config files (see below)
├── dataset/vocabs/clean/           # Training data (EN/VI sentence pairs)
├── checkpoints/                     # Model checkpoints
├── logs/                            # Training logs
├── results/                         # Experiment results
├── main.py                          # Training entry point
└── evaluate.py                      # Inference & evaluation
```

## Installation

```bash
# 1. Create and activate virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Configs

Each experiment uses a dedicated YAML config. All configs share the same data paths under `dataset/vocabs/clean/`.

| Config | Model | Tokenizer |
|--------|-------|-----------|
| `configs/transformer_bpe.yaml` | Transformer | BPE |
| `configs/transformer_phoneme.yaml` | Transformer | Phoneme |
| `configs/transformer_unigram.yaml` | Transformer | Unigram |
| `configs/lstm_seq2seq_bpe.yaml` | LSTM Seq2Seq | BPE |
| `configs/lstm_seq2seq_phoneme.yaml` | LSTM Seq2Seq | Phoneme |
| `configs/lstm_seq2seq_unigram.yaml` | LSTM Seq2Seq | Unigram |
| `configs/lstm_luong_bpe.yaml` | LSTM + Luong | BPE |
| `configs/lstm_luong_phoneme.yaml` | LSTM + Luong | Phoneme |
| `configs/lstm_luong_unigram.yaml` | LSTM + Luong | Unigram |

## Training

Run with the specific config file:

```bash
# Transformer + BPE
python main.py --config configs/transformer_bpe.yaml

# LSTM + Luong + Phoneme
python main.py --config configs/lstm_luong_phoneme.yaml

# LSTM Seq2Seq + Unigram
python main.py --config configs/lstm_seq2seq_unigram.yaml
```

Override config values via CLI arguments:

```bash
python main.py --config configs/transformer_bpe.yaml \
    --num_epochs 10 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --seed 42

# Resume from checkpoint
python main.py --config configs/transformer_bpe.yaml \
    --resume checkpoints/transformer_bpe/model_epoch_001.pt
```

## Available Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--config` | str | Path to YAML config file |
| `--num_epochs` | int | Number of training epochs |
| `--batch_size` | int | Batch size |
| `--learning_rate` | float | Learning rate |
| `--seed` | int | Random seed |
| `--src_level` | str | Source tokenization level (overrides config) |
| `--tgt_level` | str | Target tokenization level (overrides config) |
| `--max_length` | int | Max sequence length |
| `--eval_steps` | int | Evaluate every N steps |
| `--save_steps` | int | Save checkpoint every N steps |
| `--resume` | str | Path to checkpoint to resume from |

## Evaluation

```bash
python evaluate.py --checkpoint checkpoints/transformer_bpe/best_model.pt \
    --config configs/transformer_bpe.yaml
```
