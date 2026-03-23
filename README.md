# English-Vietnamese Neural Machine Translation

A research project implementing baseline Neural Machine Translation (NMT) models for English-Vietnamese translation, supporting various tokenization strategies like **BPE**, **Unigram**, **Word-level**, and **Phoneme-level** processing.

## Overview

This project provides comprehensive implementations of baseline NMT architectures for English-Vietnamese machine translation research:

- **LSTM + Bahdanau Attention**: Sequence-to-sequence model with additive attention mechanism
- **LSTM + Luong Attention**: Sequence-to-sequence model with multiplicative attention (general, dot, concat variants)
- **Transformer**: Attention-based architecture following "Attention is All You Need" (Vaswani et al., 2017)

All models support:
- Training and inference pipelines
- Comprehensive evaluation with BLEU, ROUGE, and METEOR metrics
- **BPE / Unigram processing**: Subword-level tokenization strategies
- **Word-level processing**: Traditional word-based tokenization
- **Phoneme-level processing**: Phoneme-based tokenization for both English and Vietnamese
- Autoregressive decoding with state caching
- Flexible configuration via YAML files and command-line arguments

## Project Structure

```
Phoneme_NMT/
├── src/                          # Source code
│   ├── models/                   # Model implementations
│   │   ├── base_model.py         # Abstract base model class
│   │   ├── attention/            # Attention mechanisms (for LSTM models)
│   │   │   ├── bahdanau.py       # Bahdanau Attention
│   │   │   └── luong.py          # Luong Attention
│   │   ├── lstm/                 # LSTM-based models
│   │   │   ├── encoder.py        # LSTM Encoder
│   │   │   ├── lstm_bahdanau.py  # LSTM + Bahdanau Attention
│   │   │   └── lstm_luong.py     # LSTM + Luong Attention
│   │   └── transformer/          # Transformer (modular architecture)
│   │       ├── transformer.py    # Main Transformer model
│   │       ├── encoder.py         # Transformer Encoder
│   │       ├── decoder.py         # Transformer Decoder
│   │       ├── blocks/           # Encoder/Decoder layers
│   │       │   ├── encoder_layer.py
│   │       │   └── decoder_layer.py
│   │       ├── layers/           # Core attention & feed-forward layers
│   │       │   ├── multi_head_attention.py
│   │       │   ├── scale_dot_product_attention.py
│   │       │   └── position_wise_feed_forward.py
│   │       └── embedding/        # Embedding components
│   │           ├── positional_encoding.py
│   │           ├── token_embeddings.py
│   │           └── transformer_embedding.py
│   ├── training/                 # Training infrastructure
│   │   └── trainer.py            # Main training class
│   ├── evaluation/               # Evaluation metrics
│   │   ├── evaluator.py          # Main evaluation class
│   │   ├── bleu.py               # BLEU score implementation
│   │   ├── rouge.py              # ROUGE score implementation
│   │   └── meteor.py             # METEOR score implementation
│   ├── data/                     # Data processing
│   │   ├── data_loader.py        # Data loading utilities
│   │   ├── preprocessing.py      # Data preprocessing
│   │   └── vocabs/               # Vocabulary classes
│   └── utils/                    # Utilities
│       └── logger.py             # Logging utilities
├── configs/                      # Configuration files
│   ├── config.py                 # Config management (Pydantic-based)
│   ├── lstm_bahdanau.yaml        # LSTM + Bahdanau configuration
│   ├── lstm_luong.yaml           # LSTM + Luong configuration
│   └── transformer.yaml          # Transformer configuration
├── dataset/                      # Raw data directory
│   └── vocabs/                   # Vocabulary files
├── checkpoints/                  # Model checkpoints
├── logs/                         # Training logs
├── results/                      # Experiment results
├── main.py                       # Main entry point
└── test_*.py                     # Test scripts
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/tuikhongtenbo/Phoneme_NMT.git
cd Phoneme_NMT
```

2. **Create a virtual environment** 

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

## Usage

### Training

#### Running 1 Model with 1 Tokenizer

To train a specific model with a specific tokenizer, use the `--config` flag to select the model architecture and the `--level` flag to specify the tokenizer type (e.g., `bpe`, `unigram`, `phoneme`).

```bash
# Example 1: Train Transformer model with Phoneme tokenizer
python main.py --config configs/transformer.yaml --level phoneme

# Example 2: Train LSTM + Luong Attention model with BPE tokenizer
python main.py --config configs/lstm_luong.yaml --level bpe

# Example 3: Train LSTM + Bahdanau Attention model with Unigram tokenizer
python main.py --config configs/lstm_bahdanau.yaml --level unigram
```

#### Command-Line Arguments

Override configuration parameters via command-line arguments:

```bash
# Override batch size and number of epochs
python main.py --config configs/transformer.yaml \
    --batch_size 32 \
    --num_epochs 20

# Override random seed
python main.py --config configs/transformer.yaml \
    --seed 42

# Resume training from checkpoint
python main.py --config configs/transformer.yaml \
    --resume checkpoints/transformer/model_epoch_001.pt

# Combine multiple overrides
python main.py --config configs/transformer.yaml \
    --batch_size 16 \
    --num_epochs 10 \
    --learning_rate 0.0001 \
    --level phoneme \
    --max_length 100 \
    --save_steps 2500 \
    --seed 123
```

#### Available Arguments

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--config` | str | Path to YAML configuration file | `configs/transformer.yaml` |
| `--batch_size` | int | Batch size (overrides config) | None |
| `--num_epochs` | int | Number of training epochs (overrides config) | None |
| `--learning_rate` | float | Learning rate (overrides config) | None |
| `--level` | str | Processing level/tokenizer: `bpe`, `unigram`, `phoneme`, or `word` (overrides config) | None |
| `--max_length` | int | Maximum sequence length (overrides config.data.max_seq_len) | None |
| `--eval_steps` | int | Evaluate every N steps (overrides config.training.eval_every) | None |
| `--save_steps` | int | Save checkpoint every N steps (overrides config.training.save_every) | None |
| `--seed` | int | Random seed for reproducibility (overrides config) | None |
| `--resume` | str | Path to checkpoint file to resume training | None |

**Processing Levels (Tokenizers):**
- `bpe`: BPE-based tokenization
- `unigram`: Unigram-based tokenization
- `phoneme`: Phoneme-level tokenization
- `word`: Word-level tokenization (builds vocabulary from training data)

## Evaluation Metrics

All models are evaluated using standard NMT metrics:

- **BLEU**: BLEU@1, BLEU@2, BLEU@3, BLEU@4
- **ROUGE**: ROUGE-L (Longest Common Subsequence)
- **METEOR**: METEOR score

Metrics are computed automatically during validation and can be logged for analysis. Evaluation works consistently across all processing levels (word, phoneme, and pretrained tokenizers).

