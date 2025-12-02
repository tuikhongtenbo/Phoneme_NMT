# English-Vietnamese Neural Machine Translation

A research project implementing baseline Neural Machine Translation (NMT) models for English-Vietnamese translation, supporting **word-level**, **phoneme-level**, and **pretrained tokenizer** processing.

## Overview

This project provides comprehensive implementations of baseline NMT architectures for English-Vietnamese machine translation research:

- **LSTM + Bahdanau Attention**: Sequence-to-sequence model with additive attention mechanism
- **LSTM + Luong Attention**: Sequence-to-sequence model with multiplicative attention (general, dot, concat variants)
- **Transformer**: Attention-based architecture following "Attention is All You Need" (Vaswani et al., 2017)

All models support:
- Training and inference pipelines
- Comprehensive evaluation with BLEU, ROUGE, and METEOR metrics
- **Word-level processing**: Traditional word-based tokenization
- **Phoneme-level processing**: Phoneme-based tokenization for both English and Vietnamese
- **Pretrained tokenizers**: 
  - `pretrained_1`: mBART (English) → mBART (Vietnamese)
  - `pretrained_2`: mBART (English) → BARTPho (Vietnamese)
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

#### Basic Training

Train a model using a configuration file:

```bash
# Train Transformer model
python main.py --config configs/transformer.yaml

# Train LSTM + Bahdanau model
python main.py --config configs/lstm_bahdanau.yaml

# Train LSTM + Luong model
python main.py --config configs/lstm_luong.yaml
```

#### Command-Line Arguments

Override configuration parameters via command-line arguments:

```bash
# Override batch size and number of epochs
python main.py --config configs/transformer.yaml \
    --batch_size 32 \
    --num_epochs 20

# Override processing level (word, phoneme, pretrained_1, or pretrained_2)
python main.py --config configs/transformer.yaml \
    --level word

# Use pretrained tokenizers
python main.py --config configs/transformer.yaml \
    --level pretrained_1  # mBART -> mBART

python main.py --config configs/transformer.yaml \
    --level pretrained_2  # mBART -> BARTPho

# Override random seed
python main.py --config configs/transformer.yaml \
    --seed 42

# Resume training from checkpoint
python main.py --config configs/transformer.yaml \
    --resume checkpoints/transformer/model_epoch_001.pt

# Override learning rate and training steps
python main.py --config configs/transformer.yaml \
    --learning_rate 0.0001 \
    --eval_steps 500 \
    --save_steps 2500

# Override maximum sequence length
python main.py --config configs/transformer.yaml \
    --max_length 128

# Combine multiple overrides
python main.py --config configs/transformer.yaml \
    --batch_size 16 \
    --num_epochs 10 \
    --learning_rate 0.0001 \
    --level phoneme \
    --max_length 100 \
    --eval_steps 500 \
    --save_steps 2500 \
    --seed 123

# Use pretrained tokenizers with any model
python main.py --config configs/lstm_bahdanau.yaml \
    --level pretrained_1

python main.py --config configs/lstm_luong.yaml \
    --level pretrained_2
```

#### Available Arguments

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--config` | str | Path to YAML configuration file | `configs/transformer.yaml` |
| `--batch_size` | int | Batch size (overrides config) | None |
| `--num_epochs` | int | Number of training epochs (overrides config) | None |
| `--learning_rate` | float | Learning rate (overrides config) | None |
| `--level` | str | Processing level: `word`, `phoneme`, `pretrained_1`, or `pretrained_2` (overrides config) | None |
| `--max_length` | int | Maximum sequence length (overrides config.data.max_seq_len) | None |
| `--eval_steps` | int | Evaluate every N steps (overrides config.training.eval_every) | None |
| `--save_steps` | int | Save checkpoint every N steps (overrides config.training.save_every) | None |
| `--seed` | int | Random seed for reproducibility (overrides config) | None |
| `--resume` | str | Path to checkpoint file to resume training | None |

**Processing Levels:**
- `word`: Word-level tokenization (builds vocabulary from training data)
- `phoneme`: Phoneme-level tokenization (builds phoneme vocabulary from training data)
- `pretrained_1`: Uses mBART tokenizer for both English (source) and Vietnamese (target)
- `pretrained_2`: Uses mBART tokenizer for English (source) and BARTPho tokenizer for Vietnamese (target)

## Evaluation Metrics

All models are evaluated using standard NMT metrics:

- **BLEU**: BLEU@1, BLEU@2, BLEU@3, BLEU@4
- **ROUGE**: ROUGE-L (Longest Common Subsequence)
- **METEOR**: METEOR score

Metrics are computed automatically during validation and can be logged for analysis. Evaluation works consistently across all processing levels (word, phoneme, and pretrained tokenizers).

## Pretrained Tokenizers

The project supports two pretrained tokenizer configurations:

### pretrained_1: mBART → mBART
- **Source (English)**: `facebook/mbart-large-50` with language code `en_XX`
- **Target (Vietnamese)**: `facebook/mbart-large-50` with language code `vi_VN`
- Suitable for multilingual translation scenarios

### pretrained_2: mBART → BARTPho
- **Source (English)**: `facebook/mbart-large-50` with language code `en_XX`
- **Target (Vietnamese)**: `vinai/bartpho-word` (Vietnamese-specific BART model)
- Leverages Vietnamese-specific pretraining for better target language understanding

**Usage Example:**
```bash
# Use pretrained_1 with Transformer
python main.py --config configs/transformer.yaml --level pretrained_1

# Use pretrained_2 with LSTM Bahdanau
python main.py --config configs/lstm_bahdanau.yaml --level pretrained_2
```

**Note**: When using pretrained tokenizers, the vocabulary size is determined by the tokenizer (typically 250K+ tokens for mBART). Make sure your model's embedding dimensions are compatible (e.g., 1024 for mBART-based models).

## 📚 References

1. **Transformer**: Vaswani, A., et al. (2017). "Attention is All You Need". *Advances in Neural Information Processing Systems*, 30.

2. **Bahdanau Attention**: Bahdanau, D., Cho, K., & Bengio, Y. (2015). "Neural Machine Translation by Jointly Learning to Align and Translate". *International Conference on Learning Representations*.

3. **Luong Attention**: Luong, M. T., Pham, H., & Manning, C. D. (2015). "Effective Approaches to Attention-based Neural Machine Translation". *Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing*.

4. **mBART**: Liu, Y., et al. (2020). "Multilingual Denoising Pre-training for Neural Machine Translation". *Transactions of the Association for Computational Linguistics*, 8.

5. **BARTPho**: Nguyen, V. Q., & Nguyen, T. (2021). "PhoBERT: Pre-trained language models for Vietnamese". *Findings of the Association for Computational Linguistics: EMNLP 2021*.

## 🙏 Acknowledgments

This project is part of research on English-Vietnamese Neural Machine Translation, exploring word-level, phoneme-level, and pretrained tokenizer approaches to translation.
