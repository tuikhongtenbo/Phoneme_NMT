# English-Vietnamese Machine Translation Baseline Models

This repository contains implementations of baseline models for English-Vietnamese machine translation research.

## Project Structure

```
uit-eng-vi-translation/
├── src/                       # Source code
│   ├── models/               # Model implementations
│   │   ├── attention/         # Attention mechanisms
│   │   │   ├── bahdanau.py   # Bahdanau Attention
│   │   │   ├── luong.py      # Luong Attention
│   │   │   ├── multi_head.py # Multi-head Attention
│   │   │   └── scaled_dot_product.py # Scaled Dot-Product Attention
│   │   ├── base_model.py     # Base model class
│   │   ├── lstm_bahdanau.py  # LSTM + Bahdanau Attention
│   │   ├── lstm_luong.py     # LSTM + Luong Attention
│   │   └── transformer.py    # Transformer model
│   ├── training/             # Training scripts
│   │   └── trainer.py        # Main training class
│   ├── evaluation/           # Evaluation metrics
│   │   ├── evaluator.py      # Main evaluation class
│   │   ├── metrics.py        # BLEU, ROUGE, METEOR implementations
│   │   ├── word_level.py     # Word-level evaluation
│   │   └── phoneme_level.py  # Phoneme-level evaluation
│   ├── data/                 # Data processing
│   │   ├── data_loader.py    # Data loading utilities
│   │   └── preprocessing.py  # Data preprocessing
│   └── utils/                # Utilities (empty)
├── configs/                  # Configuration files
│   ├── config.py             # Config management with dot notation
│   ├── lstm_bahdanau.yaml    # LSTM + Bahdanau config
│   ├── lstm_luong.yaml       # LSTM + Luong config
│   └── transformer.yaml      # Transformer config
├── scripts/                  # Main execution scripts (empty)
├── notebooks/                # Jupyter notebooks
│   ├── cleaner.ipynb         # Data cleaning notebook
│   ├── merging.ipynb         # Data merging notebook
│   └── test_vocab.ipynb      # Vocabulary testing notebook
├── dataset/                  # Raw data
│   └── vocabs/               # Vocabulary files
├── vocabs/                   # Additional vocabulary files
├── results/                  # Experiment results (empty)
├── logs/                     # Training logs (empty)
└── main.py                   # Main entry point
```

## Baseline Models

1. **LSTM + Bahdanau Attention**
2. **LSTM + Luong Attention**
3. **Transformer**

## Evaluation Metrics

- BLEU@1, BLEU@2, BLEU@3, BLEU@4
- ROUGE-L
- METEOR

All metrics evaluated at both **Word** and **Phoneme** levels.

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run main script
python main.py

# Run notebooks for data processing
jupyter notebook notebooks/
```

## Configuration

Use dot notation for easy config access:

```python
from configs.config import Config

# Load configuration
config = Config.from_yaml('configs/lstm_bahdanau.yaml')
```
