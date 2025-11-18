"""
Attention mechanisms for English-Vietnamese Neural Machine Translation.

"""

from .bahdanau import BahdanauAttention
from .luong import LuongAttention

__all__ = [
    'BahdanauAttention',
    'LuongAttention',
]