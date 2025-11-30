# src/data/helpers.py
"""
Helper functions for vocabulary configuration and file paths.
"""

from typing import Tuple, Any
import os

from .constants import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN


def create_vi_vocab_config(pydantic_config: Any) -> Any:
    """Create config object compatible with ViWordVocab."""
    class ViVocabConfig:
        def __init__(self, pydantic_config):
            self.TOKENIZER = "word"
            self.PAD_TOKEN = PAD_TOKEN
            self.BOS_TOKEN = SOS_TOKEN
            self.EOS_TOKEN = EOS_TOKEN
            self.UNK_TOKEN = UNK_TOKEN
            self.data = pydantic_config.data if hasattr(pydantic_config, 'data') else None
    
    return ViVocabConfig(pydantic_config)


def get_vocab_filepath(source_level: str, target_level: str, min_count: int = None, vocab_dir: str = "vocabs") -> Tuple[str, str]:
    """
    Generate vocabulary file paths based on source and target levels.
    
    Args:
        source_level: 'word' or 'phoneme' for source (EN)
        target_level: 'word' or 'phoneme' for target (VI)
        min_count: Minimum word count (for word-level vocab filtering)
        vocab_dir: Directory to save vocabularies
        
    Returns:
        Tuple of (input_vocab_path, output_vocab_path)
    """
    os.makedirs(vocab_dir, exist_ok=True)
    
    # Create filename based on levels
    source_suffix = "word" if source_level == 'word' else "phoneme"
    target_suffix = "word" if target_level == 'word' else "phoneme"
    
    filename_suffix = f"en_{source_suffix}_vi_{target_suffix}"
    if min_count and min_count > 1:
        filename_suffix += f"_min{min_count}"
    
    input_vocab_path = os.path.join(vocab_dir, f"vocab_input_{filename_suffix}.json")
    output_vocab_path = os.path.join(vocab_dir, f"vocab_output_{filename_suffix}.json")
    
    return input_vocab_path, output_vocab_path