# src/data/constants.py
"""
Special tokens and their IDs for vocabulary consistency.
"""

# Define Special Tokens and IDs for consistency
PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN = '<PAD>', '<SOS>', '<EOS>', '<UNK>'
PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3

__all__ = [
    'PAD_TOKEN', 'SOS_TOKEN', 'EOS_TOKEN', 'UNK_TOKEN',
    'PAD_ID', 'SOS_ID', 'EOS_ID', 'UNK_ID'
]