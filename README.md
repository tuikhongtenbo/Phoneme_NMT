# English-Vietnamese Neural Machine Translation

A research project implementing baseline Neural Machine Translation (NMT) models for English-Vietnamese translation, supporting both **word-level** and **phoneme-level** processing.

## Overview

This project provides comprehensive implementations of baseline NMT architectures for English-Vietnamese machine translation research:

- **LSTM + Bahdanau Attention**: Sequence-to-sequence model with additive attention mechanism
- **LSTM + Luong Attention**: Sequence-to-sequence model with multiplicative attention (general, dot, concat variants)
- **Transformer**: Attention-based architecture following "Attention is All You Need" (Vaswani et al., 2017)

All models support:
- Training and inference pipelines
- Comprehensive evaluation with BLEU, ROUGE, and METEOR metrics
- Both word-level and phoneme-level processing
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

# Override processing level (word or phoneme)
python main.py --config configs/transformer.yaml \
    --level word

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
    --level phoneme \
    --seed 123
```

#### Available Arguments

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--config` | str | Path to YAML configuration file | `configs/transformer.yaml` |
| `--batch_size` | int | Batch size (overrides config) | None |
| `--num_epochs` | int | Number of training epochs (overrides config) | None |
| `--level` | str | Processing level: `word` or `phoneme` (overrides config) | None |
| `--seed` | int | Random seed for reproducibility (overrides config) | None |
| `--resume` | str | Path to checkpoint file to resume training | None |


## Configuration

### Configuration File Structure

Each YAML configuration file follows this structure:

```yaml
model:
  name: "transformer"              # Model name: "transformer", "lstm_bahdanau", or "lstm_luong"
  embed_dim: 512                   # Embedding dimension
  hidden_dim: 512                  # Hidden dimension (for LSTM)
  num_layers: 6                    # Number of layers
  dropout: 0.1                     # Dropout rate
  num_heads: 8                     # Number of attention heads (Transformer only)
  ff_dim: 2048                     # Feed-forward dimension (Transformer only)
  attention_type: "general"        # Attention type (LSTM Luong only: "general", "dot", "concat")

training:
  batch_size: 8                    # Batch size
  num_epochs: 10                   # Number of epochs
  learning_rate: 0.001             # Learning rate
  optimizer: "adamw"               # Optimizer: "adam", "sgd", or "adamw"
  scheduler: "cosine"              # Learning rate scheduler
  clip_grad_norm: 5.0              # Gradient clipping threshold
  warmup_steps: 4000               # Warmup steps (Transformer only)
  eval_every: 1000                 # Evaluation frequency (steps)
  save_every: 5000                 # Checkpoint saving frequency (steps)

data:
  # Special token IDs
  sos_id: 1                        # Start-of-sequence token ID
  eos_id: 2                        # End-of-sequence token ID
  pad_id: 0                        # Padding token ID
  unk_id: 3                        # Unknown token ID
  
  # Processing level (must match for source and target)
  source_level: "word"              # Source level: "word" or "phoneme"
  target_level: "word"              # Target level: "word" or "phoneme" (must match source_level)
  
  # Data file paths
  train_src: "path/to/train.en"
  train_tgt: "path/to/train.vi"
  dev_src: "path/to/dev.en"
  dev_tgt: "path/to/dev.vi"
  test_src: "path/to/test.en"
  test_tgt: "path/to/test.vi"
  
  # Vocabulary settings
  vocab_json_train: "path/to/vocab.json"  # For phoneme-level processing
  min_count: 3                     # Minimum word count for vocabulary
  max_seq_len: 64                  # Maximum sequence length

device: "cuda"                     # Device: "cuda" or "cpu"
seed: 42                           # Random seed
```

### Using Configuration in Code

```python
from configs.config import Config

# Load configuration from YAML file
config = Config.from_yaml('configs/transformer.yaml')

# Access configuration values using dot notation
print(config.model.embed_dim)
print(config.training.batch_size)
print(config.data.source_level)

## Evaluation Metrics

All models are evaluated using standard NMT metrics at both **word-level** and **phoneme-level**:

- **BLEU**: BLEU@1, BLEU@2, BLEU@3, BLEU@4
- **ROUGE**: ROUGE-L (Longest Common Subsequence)
- **METEOR**: METEOR score

Metrics are computed automatically during validation and can be logged for analysis.

## Processing Levels

The project supports two processing levels:

- **Word-level**: Translation at word granularity (word → word)
- **Phoneme-level**: Translation at phoneme granularity (phoneme → phoneme)

**Important Note:** `source_level` and `target_level` must match. The project does not support mixed-level translation (e.g., word → phoneme or phoneme → word). Both source and target must use the same level.

## 📚 References

1. **Transformer**: Vaswani, A., et al. (2017). "Attention is All You Need". *Advances in Neural Information Processing Systems*, 30.

2. **Bahdanau Attention**: Bahdanau, D., Cho, K., & Bengio, Y. (2015). "Neural Machine Translation by Jointly Learning to Align and Translate". *International Conference on Learning Representations*.

3. **Luong Attention**: Luong, M. T., Pham, H., & Manning, C. D. (2015). "Effective Approaches to Attention-based Neural Machine Translation". *Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing*.

## 📄 License

[Add license information here]

## 👥 Contributors

[Add contributor information here]

## 🙏 Acknowledgments

This project is part of research on English-Vietnamese Neural Machine Translation, exploring both word-level and phoneme-level approaches to translation.
