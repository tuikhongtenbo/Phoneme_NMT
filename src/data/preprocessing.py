# src/data/preprocessing.py

from typing import List, Dict, Tuple, Any, Optional
import os
import torch
from collections import Counter

# Import utilities and Vocab class from src.utils/
from ..utils.util import preprocess_sentence
from ..utils.viword_vocab import ViWordVocab # Phoneme/Syllable-level Vocab
from ..utils.Vietnamese_util import analyze_Vietnamese 

# Define Special Tokens and IDs for consistency
PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN = '<PAD>', '<SOS>', '<EOS>', '<UNK>'
PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3


# --- Shared Base Vocab Class ---

class BaseVocab:
    """Base class for handling word-level vocabulary (Used for EN and VI Word-Level)."""
    
    def __init__(self, name: str = 'en'):
        self.name = name
        self.word2index: Dict[str, int] = {}
        self.word2count: Dict[str, int] = {}
        self.index2word: Dict[int, str] = {}
        self.count = 0
        self.specials = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
        self.initialize_special_tokens()
        
    def initialize_special_tokens(self) -> None:
        """Assign fixed IDs to special tokens."""
        self.padding_token = PAD_TOKEN
        self.bos_token = SOS_TOKEN
        self.eos_token = EOS_TOKEN
        self.unk_token = UNK_TOKEN
        
        self.padding_idx = PAD_ID
        self.bos_idx = SOS_ID
        self.eos_idx = EOS_ID
        self.unk_idx = UNK_ID

        for i, token in enumerate(self.specials):
            if token not in self.word2index:
                self.word2index[token] = i
                self.index2word[i] = token
                self.word2count[token] = 0
                self.count += 1
                
    def add_word(self, word: str):
        if word not in self.word2index:
            self.word2index[word] = self.count
            self.word2count[word] = 1
            self.index2word[self.count] = word
            self.count += 1
        else:
            self.word2count[word] += 1

    def add_sentence(self, sentence: str) -> None:
        """Tokenize and add words from a sentence."""
        words = preprocess_sentence(sentence) 
        for word in words:
            self.add_word(word)

    def trim(self, min_count: int):
        """Removes words with frequency less than min_count (excluding special tokens)."""
        initial_counts = self.word2count.copy()
        keep_words = [word for word, count in initial_counts.items() 
                      if count >= min_count or word in self.specials]
        
        # Reinitialize to reassign contiguous IDs
        self.__init__(self.name)
        
        for word in keep_words:
            if word not in self.specials:
                # Re-add word with its original count
                self.add_word(word)
                self.word2count[word] = initial_counts[word]
        
    def sentence_to_indices(self, sentence: str) -> List[int]:
        """Converts a tokenized sentence to a list of word indices (IDs)."""
        indices = []
        unk_index = self.unk_idx
        for word in preprocess_sentence(sentence):
            indices.append(self.word2index.get(word, unk_index))
        return indices
        
# --- Specific Vocab Classes ---

class EnWordVocab(BaseVocab):
    """Word-level vocabulary for English (Source). Inherits from BaseVocab."""
    pass

class ViWordLevelVocab(BaseVocab):
    """Word-level vocabulary for Vietnamese (Target). Inherits from BaseVocab."""
    
    def __init__(self, config):
        super().__init__('vi_word')
        # self.config = config # You might need config for other settings
        
    def encode_caption(self, words: List[str]) -> List[int]:
        """Converts list of Vietnamese words to list of Word IDs (1D vector)."""
        indices = [self.bos_idx]
        unk_index = self.unk_idx
        for word in words:
            # Note: preprocess_sentence is already applied in add_sentence/add_word
            # Here we assume 'words' is already a list of words from the raw VI sentence
            indices.append(self.word2index.get(word, unk_index))
        indices.append(self.eos_idx)
        return indices

# --- MAIN DATA LOADING FUNCTIONS ---

def load_pairs(data_root: str, split: str) -> List[Tuple[str, str]]:
    """Loads raw sentence pairs."""
    # (Hàm load_pairs giữ nguyên)
    base_path = os.path.join(data_root, 'tokenization', split)
    en_path = os.path.join(base_path, f'{split}.en')
    vi_path = os.path.join(base_path, f'{split}.vi')
    
    if not os.path.exists(en_path) or not os.path.exists(vi_path):
        print(f"⚠️ WARNING: Files not found at: {base_path}. Skipping.")
        return []

    print(f"Loading {split} data from: {base_path}...")
    
    with open(en_path, 'r', encoding='utf-8') as f:
        en_lines = [line.strip() for line in f if line.strip()]
    
    with open(vi_path, 'r', encoding='utf-8') as f:
        vi_lines = [line.strip() for line in f if line.strip()]
        
    if len(en_lines) != len(vi_lines):
        raise ValueError(f"Sentence count mismatch between {en_path} and {vi_path}: {len(en_lines)} vs {len(vi_lines)}")

    return list(zip(en_lines, vi_lines))


def prepare_data(data_root: str, splits: List[str], max_len: int, min_count: int = 3, config: Any = None) -> Dict[str, Any]:
    """
    Loads raw data, builds vocabularies, converts sentences to indices, and filters.
    The tokenization level for Target (VI) is selected via config.target_level.
    """
    
    # 1. Load all raw data
    all_raw_pairs = {}
    for split in splits:
        all_raw_pairs[split] = load_pairs(data_root, split)

    if 'train' not in all_raw_pairs:
        raise ValueError("Missing 'train' split required for vocabulary building.")

    # 2. Build English Vocab (Source - always Word-level)
    input_vocab = EnWordVocab("en")
    for en_sent, _ in all_raw_pairs['train']:
        input_vocab.add_sentence(en_sent)
    input_vocab.trim(min_count)
    print(f"Input Vocab Size (EN Word): {input_vocab.count}")
    
    # 3. Initialize Vietnamese Vocab (Target - level depends on config)
    target_level = getattr(config, 'target_level', 'phoneme').lower() # Default to 'phoneme'
    print(f"Target (VI) tokenization level set to: **{target_level}**")

    if target_level == 'word':
        output_vocab = ViWordLevelVocab(config)
        # Build VI Word-level vocab from train split
        for _, vi_sent in all_raw_pairs['train']:
            output_vocab.add_sentence(vi_sent)
        output_vocab.trim(min_count)
        print(f"Output Vocab Size (VI Word): {output_vocab.count}")
    
    elif target_level == 'phoneme':
        try:
            # ViWordVocab handles its own vocab building (e.g., from JSON)
            output_vocab = ViWordVocab(config)
            print(f"Output Vocab Size (VI Phonemes): {output_vocab.vocab_size}")
        except Exception as e:
            print(f"❌ ERROR: Could not initialize ViWordVocab. Falling back to Word-level.")
            print(f"Error details: {e}")
            target_level = 'word' # Fallback
            output_vocab = ViWordLevelVocab(config)
            for _, vi_sent in all_raw_pairs['train']:
                output_vocab.add_sentence(vi_sent)
            output_vocab.trim(min_count)
            print(f"Output Vocab Size (VI Word): {output_vocab.count}")
    
    else:
        raise ValueError(f"Unknown target_level: {target_level}. Must be 'word' or 'phoneme'.")

    # 4. Map to IDs and filter sentences
    indexed_data = {}
    
    for split, pairs in all_raw_pairs.items():
        current_indexed_pairs = []
        for en_sent, vi_sent in pairs:
            # Source (EN - Word Level)
            en_indices = [input_vocab.bos_idx] + input_vocab.sentence_to_indices(en_sent) + [input_vocab.eos_idx]
            
            # Target (VI)
            vi_words = vi_sent.split()
            # The structure of vi_indices depends on the level (1D List[int] or 2D List[List[int]])
            vi_indices = output_vocab.encode_caption(vi_words)
            
            # Determine target length for filtering
            if target_level == 'word':
                # vi_indices is List[int], length is len(vi_indices)
                vi_len = len(vi_indices)
            else: # 'phoneme'
                # vi_indices is List[List[int]], length is len(vi_indices)
                vi_len = len(vi_indices)
                
            # Filter sentence lengths
            if len(en_indices) <= max_len and vi_len <= max_len:
                # Store the indexed pair. Type of vi_indices: List[int] or List[List[int]]
                current_indexed_pairs.append((en_indices, vi_indices))
        
        indexed_data[split] = current_indexed_pairs
        print(f"Indexed {split} pairs: {len(current_indexed_pairs)}")
        
    return {
        'input_vocab': input_vocab,
        'output_vocab': output_vocab,
        'data': indexed_data,
        'target_level': target_level # Return the chosen level
    }# Data cleaning
