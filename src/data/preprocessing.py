# src/data/preprocessing.py

from typing import List, Dict, Tuple, Any, Optional
import os
import torch
import sys
import json
from tqdm import tqdm
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


class EnWordVocab:
    """Creates a word-level vocabulary for the English (Source) language."""
    
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

        """Uses preprocess_sentence (from src.utils.util) to tokenize and add words."""
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
    
    def save(self, filepath: str):
        """Save vocabulary to JSON file."""
        vocab_data = {
            'name': self.name,
            'word2index': self.word2index,
            'word2count': self.word2count,
            'index2word': {str(k): v for k, v in self.index2word.items()},  # Convert int keys to str for JSON
            'count': self.count,
            'specials': self.specials,
            'vocab_type': 'word_level'
        }
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        print(f"✓ Saved vocabulary to: {filepath}")
    
    @classmethod
    def load(cls, filepath: str, name: str = None):
        """Load vocabulary from JSON file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vocabulary file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        vocab = cls(name or vocab_data.get('name', 'vocab'))
        vocab.word2index = vocab_data['word2index']
        vocab.word2count = vocab_data['word2count']
        vocab.index2word = {int(k): v for k, v in vocab_data['index2word'].items()}  # Convert str keys back to int
        vocab.count = vocab_data['count']
        vocab.specials = vocab_data.get('specials', [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN])
        
        print(f"✓ Loaded vocabulary from: {filepath} (size: {vocab.count})")
        return vocab
        
# --- Specific Vocab Classes ---

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
        
        # Load English word -> IPA mapping 
        self.word_to_ipa = self.load_ipa_mapping(config)
        
        phonemes = self.make_vocab(config)
        phonemes = sorted(list(phonemes))
        
        self.itos = {i: tok for i, tok in enumerate(self.specials + phonemes)}
        self.stoi = {tok: i for i, tok in enumerate(self.specials + phonemes)}
        self.specials = [self.padding_token]
    
    def load_ipa_mapping(self, config) -> Dict[str, str]:
        """Load English word -> IPA mapping from JSON file specified in config."""        
        # Get JSON path from config
        json_path = None
        if hasattr(config, 'data'):
            json_path = getattr(config.data, 'vocab_json_train', None)
        
        if not json_path or not os.path.exists(json_path):
            # Try default paths
            default_paths = [
                'dataset/vocabs/clean/full_vocab.json'
            ]
            for path in default_paths:
                if os.path.exists(path):
                    json_path = path
                    break
        
        if not json_path or not os.path.exists(json_path):
            raise FileNotFoundError(
                f"English IPA vocabulary JSON file not found. "
                f"Please set vocab_json_train in config or place file at full_vocab.json"
            )
        
        print(f"Loading English IPA mapping from: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to word -> IPA mapping
        print("Converting to word -> IPA mapping...")
        word_to_ipa = {}
        for key, value in tqdm(data.items(), desc="Loading IPA mapping"):
            if isinstance(value, str):
                word_to_ipa[key.lower()] = value
            elif isinstance(value, dict) and 'caption' in value:
                word_to_ipa[key.lower()] = value['caption']
        
        print(f"✓ Loaded {len(word_to_ipa)} word -> IPA mappings")
        return word_to_ipa
    
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
        """Build phoneme vocabulary from IPA mappings in JSON file."""
        phonemes = set()
        
        # Add all possible English phonemes
        phonemes.update(EnglishIPA.Vowels)
        phonemes.update(EnglishIPA.Consonants)
        
        # Collect phonemes from IPA mappings in JSON file
        if hasattr(self, 'word_to_ipa'):
            print("Building phoneme vocabulary from IPA mappings...")
            for ipa_str in tqdm(self.word_to_ipa.values(), desc="Processing IPA strings"):
                try:
                    phoneme_seqs = convert_English_IPA_to_phoneme(ipa_str)
                    for phoneme_seq in phoneme_seqs:
                        if isinstance(phoneme_seq, tuple) and len(phoneme_seq) == 3:
                            initial, vowel, final = phoneme_seq
                            # Add each non-empty string phoneme
                            for phoneme in [initial, vowel, final]:
                                if phoneme and isinstance(phoneme, str):
                                    phonemes.add(phoneme)
                except:
                    pass  
        
        print(f"✓ Built phoneme vocabulary with {len(phonemes)} unique phonemes")
        return phonemes
    
    def encode_caption(self, sentence: str) -> List[List[int]]:
        """
        Convert English sentence to phoneme indices.
        
        Process: English words -> lookup IPA from JSON (via config) -> convert IPA to phonemes -> encode to indices
        """
        from .vocabs.utils import preprocess_sentence
        
        words = preprocess_sentence(sentence)
        encoded = [[self.bos_idx]]
        
        for word in words:
            # Lookup IPA from JSON mapping
            ipa_str = self.word_to_ipa.get(word.lower(), None)
            
            if ipa_str:
                try:
                    # Convert IPA string to phonemes
                    phoneme_seqs = convert_English_IPA_to_phoneme(ipa_str)
                    for phoneme_seq in phoneme_seqs:
                        if isinstance(phoneme_seq, tuple) and len(phoneme_seq) == 3:
                            initial, vowel, final = phoneme_seq
                            # Encode each non-empty phoneme
                            for phoneme in [initial, vowel, final]:
                                if phoneme and isinstance(phoneme, str):
                                    encoded.append([self.stoi.get(phoneme, self.unk_idx)])
                except:
                    # If IPA conversion fails, use UNK
                    encoded.append([self.unk_idx])
            else:
                # Word not found in mapping, use UNK
                encoded.append([self.unk_idx])
        
        encoded.append([self.eos_idx])
        return encoded
    
    def sentence_to_indices(self, sentence: str) -> List[int]:
        """Convert sentence to indices (for compatibility with word-level)."""
        phoneme_seqs = self.encode_caption(sentence)
        return [idx for seq in phoneme_seqs for idx in seq]
    
    def save(self, filepath: str):
        """Save phoneme vocabulary to JSON file."""
        vocab_data = {
            'itos': self.itos,
            'stoi': self.stoi,
            'specials': self.specials,
            'word_to_ipa': self.word_to_ipa if hasattr(self, 'word_to_ipa') else {},
            'vocab_type': 'phoneme_level',
            'vocab_size': self.vocab_size
        }
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        print(f"✓ Saved phoneme vocabulary to: {filepath}")
    
    @classmethod
    def load(cls, filepath: str, config: Any = None):
        """Load phoneme vocabulary from JSON file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vocabulary file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        vocab = cls.__new__(cls)  # Create instance without calling __init__
        vocab.itos = {int(k): v for k, v in vocab_data['itos'].items()}  # Convert str keys back to int
        vocab.stoi = vocab_data['stoi']
        vocab.specials = vocab_data.get('specials', [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN])
        
        # Initialize special token indices
        vocab.initialize_special_tokens()
        
        # Load word_to_ipa if available
        if 'word_to_ipa' in vocab_data and vocab_data['word_to_ipa']:
            vocab.word_to_ipa = vocab_data['word_to_ipa']
        elif config:
            # Try to load from config if not in saved data
            try:
                # Create a temporary instance to use load_ipa_mapping method
                temp_vocab = cls(config)
                vocab.word_to_ipa = temp_vocab.word_to_ipa
            except:
                vocab.word_to_ipa = {}
        else:
            vocab.word_to_ipa = {}
        
        print(f"✓ Loaded phoneme vocabulary from: {filepath} (size: {vocab.vocab_size})")
        return vocab


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
            
            # Pass the full config so ViWordVocab can access Vietnamese text file paths
            # Vietnamese phonemes are extracted from train_tgt, dev_tgt, test_tgt files
            self.data = pydantic_config.data if hasattr(pydantic_config, 'data') else None
    
    return ViVocabConfig(pydantic_config)


# --- Vocabulary File Path Helpers ---

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
    
# --- MAIN DATA LOADING FUNCTIONS ---

def load_pairs(data_root: str, split: str) -> List[Tuple[str, str]]:
    """
    Loads raw sentence pairs from the specified file structure: 
    [data_root]/tokenization/[split]/{split}.en & {split}.vi
    """
    base_path = os.path.join(data_root, 'tokenization', split)
    en_path = os.path.join(base_path, f'{split}.en')
    vi_path = os.path.join(base_path, f'{split}.vi')
    
    if not os.path.exists(en_path) or not os.path.exists(vi_path):
        print(f"WARNING: Files not found at: {base_path}. Skipping.")
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
    Loads raw data, builds vocabularies (EN word-level, VI phoneme-level), 
    converts sentences to indices, and filters based on max_len.
    """
    
    # 1. Load all raw data
    all_raw_pairs = {}
    for split in splits:
        all_raw_pairs[split] = load_pairs(data_root, split)

    if 'train' not in all_raw_pairs:
        raise ValueError("Missing 'train' split required for vocabulary building.")

    # 2. Build English Vocab (Word-level)
    input_vocab = EnWordVocab("en")
    
    # Use only TRAIN split to build vocab
    for en_sent, _ in all_raw_pairs['train']:
        input_vocab.add_sentence(en_sent)

    input_vocab.trim(min_count)
    print(f"Input Vocab Size (EN): {input_vocab.count}")
    
    # 3. Initialize Vietnamese Vocab (Phoneme/Syllable-level)
    # ViWordVocab is expected to handle its own vocabulary building (e.g., from JSON)
    try:
        vi_vocab = ViWordVocab(config)
        print(f"Output Vocab Size (VI Phonemes): {vi_vocab.vocab_size}")
    except Exception as e:
        print(f"ERROR: Could not initialize ViWordVocab. Check Config and JSON paths.")
        print(f"Error details: {e}")
        # Create a dummy Vocab to prevent crash if ViWordVocab fails to init
        class DummyViWordVocab:
             padding_idx = PAD_ID; bos_idx = SOS_ID; eos_idx = EOS_ID; unk_idx = UNK_ID
             def encode_caption(self, words): return [[self.bos_idx, PAD_ID, PAD_ID, PAD_ID], [self.unk_idx, PAD_ID, PAD_ID, PAD_ID], [self.eos_idx, PAD_ID, PAD_ID, PAD_ID]]
             vocab_size = 4
        vi_vocab = DummyViWordVocab()


    # 4. Map to IDs and filter sentences
    indexed_data = {}
    
    for split, pairs in all_raw_pairs.items():
        current_indexed_pairs = []
        for en_sent, vi_sent in pairs:
            # Source (EN - Word Level)
            # Add BOS/EOS tokens manually for sequence models
            en_indices = [input_vocab.bos_idx] + input_vocab.sentence_to_indices(en_sent) + [input_vocab.eos_idx]
            
            # Target (VI - Phoneme/Syllable Level)
            vi_words = vi_sent.split()
            # encode_caption returns a List[List[int]] (syllable, 4 components)
            vi_indices_list = vi_vocab.encode_caption(vi_words).tolist()
            
            # Filter sentence lengths (including BOS/EOS)
            if len(en_indices) <= max_len and len(vi_indices_list) <= max_len:
                # Store as List[int] and List[List[int]]
                current_indexed_pairs.append((en_indices, vi_indices_list))
        
        indexed_data[split] = current_indexed_pairs
        print(f"Indexed {split} pairs: {len(current_indexed_pairs)}")
        
    return {
        'input_vocab': input_vocab,
        'output_vocab': vi_vocab,
        'data': indexed_data
    }