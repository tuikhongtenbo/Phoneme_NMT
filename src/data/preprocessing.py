# src/data/preprocessing.py

from typing import List, Dict, Tuple, Any, Optional
import os
import torch
from collections import Counter

from .vocabs.utils import preprocess_sentence
from .vocabs.viword_vocab import ViWordVocab
from .vocabs.Vietnamese_utils import analyze_Vietnamese
from .vocabs.English_utils import convert_English_IPA_to_phoneme, EnglishIPA


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
    
    @property
    def vocab_size(self):
        """Return vocabulary size for compatibility with ViWordVocab interface."""
        return self.count
        
    def encode_caption(self, words: List[str]) -> List[int]:
        """Converts list of Vietnamese words to list of Word IDs (1D vector)."""
        indices = [self.bos_idx]
        unk_index = self.unk_idx
        for word in words:
            indices.append(self.word2index.get(word, unk_index))
        indices.append(self.eos_idx)
        return indices


class EnPhonemeVocab:
    """Phoneme-level vocabulary for English (Source)."""
    
    def __init__(self, config):
        self.initialize_special_tokens()
        phonemes = self.make_vocab(config)
        phonemes = sorted(list(phonemes))
        
        self.itos = {i: tok for i, tok in enumerate(self.specials + phonemes)}
        self.stoi = {tok: i for i, tok in enumerate(self.specials + phonemes)}
        self.specials = [self.padding_token]
    
    @property
    def vocab_size(self):
        """Return vocabulary size."""
        return len(self.itos)
    
    def initialize_special_tokens(self) -> None:
        """Initialize special tokens."""
        self.padding_token = PAD_TOKEN
        self.bos_token = SOS_TOKEN
        self.eos_token = EOS_TOKEN
        self.unk_token = UNK_TOKEN
        
        self.specials = [self.padding_token, self.bos_token, self.eos_token, self.unk_token]
        self.padding_idx = 0
        self.bos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3
    
    def make_vocab(self, config) -> set:
        """Build phoneme vocabulary from training data."""
        
        phonemes = set()
        
        # Add all possible English phonemes
        phonemes.update(EnglishIPA.Vowels)
        phonemes.update(EnglishIPA.Consonants)
        
        # Collect additional phonemes from training data if available
        if hasattr(config, 'data') and hasattr(config.data, 'train_src'):
            train_path = config.data.train_src
            if os.path.exists(train_path):
                with open(train_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            # Assume line is IPA or can be converted
                            # For now, just collect from existing phonemes
                            # Can be extended to parse IPA from data
                            pass
        
        return phonemes
    
    def encode_caption(self, sentence: str) -> List[List[int]]:
        """Convert English sentence to phoneme indices."""
        # Assume sentence is in IPA format
        try:
            phoneme_seqs = convert_English_IPA_to_phoneme(sentence)
        except:
            # If conversion fails, treat as word-level and use UNK
            phoneme_seqs = []
        
        encoded = [[self.bos_idx]]
        for phoneme_seq in phoneme_seqs:
            if isinstance(phoneme_seq, list):
                # phoneme_seq is [initial, vowel, final]
                for phoneme in phoneme_seq:
                    if phoneme:
                        encoded.append([self.stoi.get(phoneme, self.unk_idx)])
            else:
                encoded.append([self.stoi.get(phoneme_seq, self.unk_idx)])
        encoded.append([self.eos_idx])
        
        return encoded
    
    def sentence_to_indices(self, sentence: str) -> List[int]:
        """Convert sentence to indices (for compatibility with word-level)."""
        phoneme_seqs = self.encode_caption(sentence)
        return [idx for seq in phoneme_seqs for idx in seq]


# --- Config Helpers ---

def create_vi_vocab_config(pydantic_config: Any) -> Any:
    """Create config object compatible with ViWordVocab."""
    class ViVocabConfig:
        def __init__(self, pydantic_config):
            self.TOKENIZER = "word"
            self.PAD_TOKEN = PAD_TOKEN
            self.BOS_TOKEN = SOS_TOKEN
            self.EOS_TOKEN = EOS_TOKEN
            self.UNK_TOKEN = UNK_TOKEN
            
            if hasattr(pydantic_config, 'data'):
                train_path = getattr(pydantic_config.data, 'vocab_json_train', 'dataset/vocabs/full_vocab_ipa.json')
                dev_path = getattr(pydantic_config.data, 'vocab_json_dev', 'dataset/vocabs/full_vocab_ipa.json')
                test_path = getattr(pydantic_config.data, 'vocab_json_test', 'dataset/vocabs/full_vocab_ipa.json')
            else:
                train_path = 'dataset/vocabs/clean/full_vocab.json'
                dev_path = 'dataset/vocabs/clean/full_vocab.json'
                test_path = 'dataset/vocabs/clean/full_vocab.json'
            
            self.JSON_PATH = type('JSON_PATH', (), {
                'TRAIN': train_path,
                'DEV': dev_path,
                'TEST': test_path
            })()
    
    return ViVocabConfig(pydantic_config)


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

    # 3. Build Source Vocabulary (EN)
    if source_level == 'word':
        input_vocab = EnWordVocab("en")
        for en_sent, _ in all_raw_pairs['train']:
            input_vocab.add_sentence(en_sent)
        input_vocab.trim(min_count)
        print(f"Input Vocab Size (EN Word): {input_vocab.count}")
    elif source_level == 'phoneme':
        input_vocab = EnPhonemeVocab(config)
        print(f"Input Vocab Size (EN Phonemes): {input_vocab.vocab_size}")
    else:
        raise ValueError(f"Unknown source_level: {source_level}. Must be 'word' or 'phoneme'.")

    # 4. Build Target Vocabulary (VI)
    if target_level == 'word':
        output_vocab = ViWordLevelVocab(config)
        for _, vi_sent in all_raw_pairs['train']:
            output_vocab.add_sentence(vi_sent)
        output_vocab.trim(min_count)
        print(f"Output Vocab Size (VI Word): {output_vocab.count}")
    elif target_level == 'phoneme':
        try:
            vi_vocab_config = create_vi_vocab_config(config)
            output_vocab = ViWordVocab(vi_vocab_config)
            print(f"Output Vocab Size (VI Phonemes): {output_vocab.vocab_size}")
        except Exception as e:
            print(f"❌ ERROR: Could not initialize ViWordVocab. Falling back to Word-level.")
            print(f"Error details: {e}")
            target_level = 'word'
            output_vocab = ViWordLevelVocab(config)
            for _, vi_sent in all_raw_pairs['train']:
                output_vocab.add_sentence(vi_sent)
            output_vocab.trim(min_count)
            print(f"Output Vocab Size (VI Word): {output_vocab.count}")
    else:
        raise ValueError(f"Unknown target_level: {target_level}. Must be 'word' or 'phoneme'.")

    # 5. Convert to indices and filter
    indexed_data = {}
    
    for split, pairs in all_raw_pairs.items():
        current_indexed_pairs = []
        for en_sent, vi_sent in pairs:
            # Source (EN)
            if source_level == 'word':
                en_indices = [input_vocab.bos_idx] + input_vocab.sentence_to_indices(en_sent) + [input_vocab.eos_idx]
            else:  # phoneme
                en_indices = input_vocab.encode_caption(en_sent)
            
            # Target (VI)
            vi_words = preprocess_sentence(vi_sent)
            vi_indices = output_vocab.encode_caption(vi_words)
            
            # Calculate lengths for filtering
            if source_level == 'word':
                en_len = len(en_indices)
            else:  # phoneme
                en_len = len(en_indices) if isinstance(en_indices[0], list) else len(en_indices)
            
            if target_level == 'word':
                vi_len = len(vi_indices)
            else:  # phoneme
                vi_len = len(vi_indices) if isinstance(vi_indices[0], list) else len(vi_indices)
            
            # Filter by length
            if en_len <= max_len and vi_len <= max_len:
                current_indexed_pairs.append((en_indices, vi_indices))
        
        indexed_data[split] = current_indexed_pairs
        print(f"Indexed {split} pairs: {len(current_indexed_pairs)}")
        
    return {
        'input_vocab': input_vocab,
        'output_vocab': output_vocab,
        'data': indexed_data,
        'source_level': source_level,
        'target_level': target_level
    }