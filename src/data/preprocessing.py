# src/data/preprocessing.py
"""
Main data preprocessing functions for loading and preparing translation data.
"""

from typing import List, Dict, Tuple, Any
import os
import torch
import sys
from tqdm import tqdm

# Import constants
from .constants import PAD_ID

# Import vocab classes
from .vocabs.en_vocabs import EnWordVocab, EnPhonemeVocab
from .vocabs.vi_vocabs import ViWordLevelVocab
from .vocabs.utils import preprocess_sentence
from .vocabs.viword_vocab import ViWordVocab

# Import helpers
from .helpers import create_vi_vocab_config, get_vocab_filepath


# --- MAIN DATA LOADING FUNCTIONS ---

def load_pairs(split: str, config: Any) -> List[Tuple[str, str]]:
    """
    Loads raw sentence pairs using direct paths from config.
    
    Args:
        split: Data split name ('train', 'dev', or 'test')
        config: Config object with data paths
        
    Returns:
        List of (source, target) sentence pairs
    """
    # Get direct paths from config
    if not config or not hasattr(config, 'data'):
        raise ValueError("Config object with data paths is required")
    
    data_config = config.data
    
    # Determine which paths to use based on split
    if split == 'train':
        en_path = data_config.train_src
        vi_path = data_config.train_tgt
    elif split == 'dev':
        en_path = data_config.dev_src
        vi_path = data_config.dev_tgt
    elif split == 'test':
        en_path = data_config.test_src
        vi_path = data_config.test_tgt
    else:
        raise ValueError(f"Unknown split: {split}. Must be 'train', 'dev', or 'test'")
    
    # Check if files exist
    if not os.path.exists(en_path):
        raise FileNotFoundError(f"Source file not found: {en_path}")
    if not os.path.exists(vi_path):
        raise FileNotFoundError(f"Target file not found: {vi_path}")
    
    print(f"Loading {split} data:")
    print(f"  Source: {en_path}")
    print(f"  Target: {vi_path}")
    
    # Load data
    with open(en_path, 'r', encoding='utf-8') as f:
        en_lines = [line.strip() for line in f if line.strip()]
    
    with open(vi_path, 'r', encoding='utf-8') as f:
        vi_lines = [line.strip() for line in f if line.strip()]
        
    if len(en_lines) != len(vi_lines):
        raise ValueError(f"Sentence count mismatch between {en_path} and {vi_path}: {len(en_lines)} vs {len(vi_lines)}")

    print(f"✓ Loaded {len(en_lines)} sentence pairs from {split} split")
    return list(zip(en_lines, vi_lines))


def prepare_data(splits: List[str], max_len: int, min_count: int = 3, config: Any = None) -> Dict[str, Any]:
    """
    Load data, build vocabularies, convert to indices, and filter.
    
    Supports:
    - Source level: 'word' or 'phoneme' (EN)
    - Target level: 'word' or 'phoneme' (VI)
    """
    
    # 1. Load raw data
    all_raw_pairs = {}
    for split in splits:
        all_raw_pairs[split] = load_pairs(split, config)

    if 'train' not in all_raw_pairs:
        raise ValueError("Missing 'train' split required for vocabulary building.")

    # 2. Get tokenization levels from config
    if config and hasattr(config, 'data'):
        source_level = getattr(config.data, 'source_level', 'word').lower()
        target_level = getattr(config.data, 'target_level', 'word').lower()
    else:
        source_level = 'word'
        target_level = 'phoneme'
    
    print(f"Source (EN) tokenization level: **{source_level}**")
    print(f"Target (VI) tokenization level: **{target_level}**")

    # Get vocabulary file paths
    input_vocab_path, output_vocab_path = get_vocab_filepath(
        source_level, target_level, min_count
    )

    # 3. Build or Load Source Vocabulary (EN)
    if os.path.exists(input_vocab_path):
        print(f"\nFound saved input vocabulary: {input_vocab_path}")
        try:
            if source_level == 'word':
                input_vocab = EnWordVocab.load(input_vocab_path, name="en")
            else:  # phoneme
                input_vocab = EnPhonemeVocab.load(input_vocab_path, config)
            print(f"✓ Loaded input vocabulary (size: {input_vocab.count if hasattr(input_vocab, 'count') else input_vocab.vocab_size})")
        except Exception as e:
            print(f"⚠ Error loading vocabulary: {e}. Rebuilding...")
            if source_level == 'word':
                input_vocab = EnWordVocab("en")
                for en_sent, _ in all_raw_pairs['train']:
                    input_vocab.add_sentence(en_sent)
                input_vocab.trim(min_count)
                input_vocab.save(input_vocab_path)
                print(f"Input Vocab Size (EN Word): {input_vocab.count}")
            else:  # phoneme
                input_vocab = EnPhonemeVocab(config)
                input_vocab.save(input_vocab_path)
                print(f"Input Vocab Size (EN Phonemes): {input_vocab.vocab_size}")
    else:
        print(f"\nBuilding input vocabulary (will save to: {input_vocab_path})")
        if source_level == 'word':
            input_vocab = EnWordVocab("en")
            for en_sent, _ in all_raw_pairs['train']:
                input_vocab.add_sentence(en_sent)
            input_vocab.trim(min_count)
            input_vocab.save(input_vocab_path)
            print(f"Input Vocab Size (EN Word): {input_vocab.count}")
        elif source_level == 'phoneme':
            input_vocab = EnPhonemeVocab(config)
            input_vocab.save(input_vocab_path)
            print(f"Input Vocab Size (EN Phonemes): {input_vocab.vocab_size}")
        else:
            raise ValueError(f"Unknown source_level: {source_level}. Must be 'word' or 'phoneme'.")

    # 4. Build or Load Target Vocabulary (VI)
    if os.path.exists(output_vocab_path):
        print(f"\nFound saved output vocabulary: {output_vocab_path}")
        try:
            if target_level == 'word':
                output_vocab = ViWordLevelVocab.load(output_vocab_path, name='vi_word')
            else:  # phoneme
                vi_vocab_config = create_vi_vocab_config(config)
                output_vocab = ViWordVocab.load(output_vocab_path, vi_vocab_config)
            print(f"✓ Loaded output vocabulary (size: {output_vocab.count if hasattr(output_vocab, 'count') else output_vocab.vocab_size})")
        except Exception as e:
            print(f"⚠ Error loading vocabulary: {e}. Rebuilding...")
            if target_level == 'word':
                output_vocab = ViWordLevelVocab(config)
                for _, vi_sent in all_raw_pairs['train']:
                    output_vocab.add_sentence(vi_sent)
                output_vocab.trim(min_count)
                output_vocab.save(output_vocab_path)
                print(f"Output Vocab Size (VI Word): {output_vocab.count}")
            else:  # phoneme
                try:
                    vi_vocab_config = create_vi_vocab_config(config)
                    output_vocab = ViWordVocab(vi_vocab_config)
                    output_vocab.save(output_vocab_path)
                    print(f"Output Vocab Size (VI Phonemes): {output_vocab.vocab_size}")
                except Exception as e2:
                    print(f"⚠ ERROR: Could not initialize ViWordVocab. Falling back to Word-level.")
                    print(f"Error details: {e2}")
                    target_level = 'word'
                    output_vocab = ViWordLevelVocab(config)
                    for _, vi_sent in all_raw_pairs['train']:
                        output_vocab.add_sentence(vi_sent)
                    output_vocab.trim(min_count)
                    output_vocab.save(output_vocab_path)
                    print(f"Output Vocab Size (VI Word): {output_vocab.count}")
    else:
        print(f"\nBuilding output vocabulary (will save to: {output_vocab_path})")
        if target_level == 'word':
            output_vocab = ViWordLevelVocab(config)
            for _, vi_sent in all_raw_pairs['train']:
                output_vocab.add_sentence(vi_sent)
            output_vocab.trim(min_count)
            output_vocab.save(output_vocab_path)
            print(f"Output Vocab Size (VI Word): {output_vocab.count}")
        elif target_level == 'phoneme':
            try:
                vi_vocab_config = create_vi_vocab_config(config)
                output_vocab = ViWordVocab(vi_vocab_config)
                output_vocab.save(output_vocab_path)
                print(f"Output Vocab Size (VI Phonemes): {output_vocab.vocab_size}")
            except Exception as e:
                print(f"⚠ ERROR: Could not initialize ViWordVocab. Falling back to Word-level.")
                print(f"Error details: {e}")
                target_level = 'word'
                output_vocab = ViWordLevelVocab(config)
                for _, vi_sent in all_raw_pairs['train']:
                    output_vocab.add_sentence(vi_sent)
                output_vocab.trim(min_count)
                output_vocab.save(output_vocab_path)
                print(f"Output Vocab Size (VI Word): {output_vocab.count}")
        else:
            raise ValueError(f"Unknown target_level: {target_level}. Must be 'word' or 'phoneme'.")

    # 5. Convert to indices and filter
    print("\nConverting sentences to indices...")
    sys.stdout.flush()  # Ensure output is visible immediately
    
    indexed_data = {}
    
    for split, pairs in all_raw_pairs.items():
        print(f"\nProcessing {split} split ({len(pairs):,} pairs)...")
        sys.stdout.flush()
        current_indexed_pairs = []
        
        # Use tqdm with explicit file parameter for better compatibility
        progress_bar = tqdm(pairs, desc=f"Encoding {split}", file=sys.stdout, mininterval=1.0, ncols=100)
        
        for idx, (en_sent, vi_sent) in enumerate(progress_bar):
            try:
                # Source (EN)
                if source_level == 'word':
                    en_indices = [input_vocab.bos_idx] + input_vocab.sentence_to_indices(en_sent) + [input_vocab.eos_idx]
                else:  # phoneme
                    en_indices = input_vocab.encode_caption(en_sent)
                
                # Target (VI)
                vi_words = preprocess_sentence(vi_sent)
                vi_indices = output_vocab.encode_caption(vi_words)
                
                # Handle tensor return type for Vietnamese phonemes
                if isinstance(vi_indices, torch.Tensor):
                    vi_indices = vi_indices.tolist()
                
                # Calculate lengths for filtering
                if source_level == 'word':
                    en_len = len(en_indices)
                else:  # phoneme
                    # Handle nested list structure for phonemes
                    if isinstance(en_indices, list) and len(en_indices) > 0:
                        if isinstance(en_indices[0], list):
                            en_len = sum(len(item) if isinstance(item, list) else 1 for item in en_indices)
                        else:
                            en_len = len(en_indices)
                    else:
                        en_len = 0
                
                if target_level == 'word':
                    vi_len = len(vi_indices)
                else:  # phoneme
                    # Handle nested list structure for phonemes
                    if isinstance(vi_indices, list) and len(vi_indices) > 0:
                        if isinstance(vi_indices[0], list):
                            vi_len = len(vi_indices)
                        else:
                            vi_len = len(vi_indices)
                    elif isinstance(vi_indices, torch.Tensor):
                        vi_len = vi_indices.shape[0] if len(vi_indices.shape) > 0 else 0
                    else:
                        vi_len = 0
                
                # Filter by length
                if en_len <= max_len and vi_len <= max_len:
                    current_indexed_pairs.append((en_indices, vi_indices))
                
                # Update progress bar every 1000 items
                if (idx + 1) % 1000 == 0:
                    progress_bar.set_postfix({
                        'processed': f'{idx+1:,}/{len(pairs):,}',
                        'kept': f'{len(current_indexed_pairs):,}'
                    })
                    
            except Exception as e:
                if idx < 10:  
                    print(f"Warning: Error processing pair {idx}: {e}")
                continue
        
        progress_bar.close()
        indexed_data[split] = current_indexed_pairs
        print(f"✓ Indexed {split}: {len(current_indexed_pairs):,} pairs kept out of {len(pairs):,} total")
        sys.stdout.flush()
    
    print("\n" + "="*60)
    print("Data preparation completed!")
    print("="*60)
    total_pairs = sum(len(pairs) for pairs in indexed_data.values())
    print(f"Total indexed pairs across all splits: {total_pairs:,}")
    for split, pairs in indexed_data.items():
        print(f"  {split}: {len(pairs):,} pairs")
    print("="*60 + "\n")
    sys.stdout.flush()
    
    return {
        'input_vocab': input_vocab,
        'output_vocab': output_vocab,
        'data': indexed_data,
        'source_level': source_level,
        'target_level': target_level
    }