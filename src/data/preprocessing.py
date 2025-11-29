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
from .vocabs.huggingface_vocab import HuggingFaceVocab
from transformers import AutoTokenizer

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
        
        vocab = cls.__new__(cls)  # Create instance without calling _init_
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

def load_pretrained_tokenizers(config):
    """
    Load pretrained tokenizers depending on:
      - pretrained_1: mBART → mBART
      - pretrained_2: mBART → BARTPho
    """

    mode = getattr(config, "pretrained_mode", None)

    if mode is None:
        return None, None  # không dùng pretrained tokenizers

    if mode == "pretrained_1":
        print("Using pretrained_1: mBART → mBART tokenizer")
        en_tok = AutoTokenizer.from_pretrained("facebook/mbart-large-50")
        vi_tok = AutoTokenizer.from_pretrained("facebook/mbart-large-50")

    elif mode == "pretrained_2":
        print("Using pretrained_2: mBART → BARTPho tokenizer")
        en_tok = AutoTokenizer.from_pretrained("facebook/mbart-large-50")
        vi_tok = AutoTokenizer.from_pretrained("vinai/bartpho-syllable")

    else:
        raise ValueError(f"Unknown pretrained mode: {mode}")

    return en_tok, vi_tok

def prepare_data(splits: List[str], max_len: int, min_count: int = 3, config: Any = None) -> Dict[str, Any]:
    """
    Load data, build vocabularies, convert to indices, and filter.
    
    Supports:
    - Source level: 'word' or 'phoneme' (EN)
    - Target level: 'word' or 'phoneme' (VI)
    """
    
    # 1. Load raw data
    # Load pretrained tokenizers if required
````en_tok, vi_tok = load_pretrained_tokenizers(config)

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
    
    print(f"Source (EN) tokenization level: *{source_level}*")
    print(f"Target (VI) tokenization level: *{target_level}*")

    # Check if using HuggingFace tokenizers
    encoder_tokenizer = None
    decoder_tokenizer = None
    if config and hasattr(config, 'data'):
        encoder_tokenizer = getattr(config.data, 'encoder_tokenizer', None)
        decoder_tokenizer = getattr(config.data, 'decoder_tokenizer', None)
    
    if encoder_tokenizer:
        print(f"Using HuggingFace tokenizer for source: {encoder_tokenizer}")
    if decoder_tokenizer:
        print(f"Using HuggingFace tokenizer for target: {decoder_tokenizer}")

    # Get vocabulary file paths
    input_vocab_path, output_vocab_path = get_vocab_filepath(
        source_level, target_level, min_count
    )
    
    # Add tokenizer suffix to vocab paths if using HuggingFace tokenizers
    if encoder_tokenizer:
        tokenizer_suffix = encoder_tokenizer.replace('/', '').replace('-', '')
        base_path = os.path.splitext(input_vocab_path)[0]
        input_vocab_path = f"{base_path}_tokenizer_{tokenizer_suffix}.json"
    if decoder_tokenizer:
        tokenizer_suffix = decoder_tokenizer.replace('/', '').replace('-', '')
        base_path = os.path.splitext(output_vocab_path)[0]
        output_vocab_path = f"{base_path}_tokenizer_{tokenizer_suffix}.json"

    # 3. Build or Load Source Vocabulary (EN)
    if encoder_tokenizer:
        # Use HuggingFace tokenizer
        print(f"\nUsing HuggingFace tokenizer for source: {encoder_tokenizer}")
        if os.path.exists(input_vocab_path):
            print(f"Found saved tokenizer config: {input_vocab_path}")
            try:
                input_vocab = HuggingFaceVocab.load(input_vocab_path)
                print(f"✓ Loaded tokenizer vocab (size: {input_vocab.vocab_size})")
            except Exception as e:
                print(f"⚠ Error loading tokenizer config: {e}. Creating new...")
                input_vocab = HuggingFaceVocab(encoder_tokenizer, lang_code='en_XX')
                input_vocab.save(input_vocab_path)
                print(f"Source Vocab Size (EN HuggingFace): {input_vocab.vocab_size}")
        else:
            print(f"\nCreating HuggingFace tokenizer vocab (will save to: {input_vocab_path})")
            input_vocab = HuggingFaceVocab(encoder_tokenizer, lang_code='en_XX')
            input_vocab.save(input_vocab_path)
            print(f"Source Vocab Size (EN HuggingFace): {input_vocab.vocab_size}")
    else:
        # Use traditional vocab building
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
                    # === ENGLISH TOKENIZATION HANDLING ===
                    if en_tok is None:
                        # --- OLD MODE: dùng vocab word-level ---
                        input_vocab = EnWordVocab("en")

                        for en_sent, _ in all_raw_pairs['train']:
                            input_vocab.add_sentence(en_sent)

                        input_vocab.trim(min_count)
                        print(f"Input Vocab Size (EN word-level): {input_vocab.count}")

                    else:
                        # --- PRETRAINED MODE (mBART tokenizer) ---
                        print("Using pretrained tokenizer for EN (mBART)")
                        input_vocab = en_tok
                    input_vocab.save(input_vocab_path)
                    print(f"Input Vocab Size (EN Word): {input_vocab.count}")
                else:  # phoneme
                    input_vocab = EnPhonemeVocab(config)
                    input_vocab.save(input_vocab_path)
                    print(f"Input Vocab Size (EN Phonemes): {input_vocab.vocab_size}")
        else:
            print(f"\nBuilding input vocabulary (will save to: {input_vocab_path})")
            if source_level == 'word':
                if en_tok is None:
                        # --- OLD MODE: dùng vocab word-level ---
                        input_vocab = EnWordVocab("en")

                        for en_sent, _ in all_raw_pairs['train']:
                            input_vocab.add_sentence(en_sent)

                        input_vocab.trim(min_count)
                        print(f"Input Vocab Size (EN word-level): {input_vocab.count}")

                    else:
                        # --- PRETRAINED MODE (mBART tokenizer) ---
                        print("Using pretrained tokenizer for EN (mBART)")
                        input_vocab = en_tok
                input_vocab.save(input_vocab_path)
                print(f"Input Vocab Size (EN Word): {input_vocab.count}")
            elif source_level == 'phoneme':
                input_vocab = EnPhonemeVocab(config)
                input_vocab.save(input_vocab_path)
                print(f"Input Vocab Size (EN Phonemes): {input_vocab.vocab_size}")
            else:
                raise ValueError(f"Unknown source_level: {source_level}. Must be 'word' or 'phoneme'.")

    # 4. Build or Load Target Vocabulary (VI)
    if decoder_tokenizer:
        # Use HuggingFace tokenizer
        print(f"\nUsing HuggingFace tokenizer for target: {decoder_tokenizer}")
        if os.path.exists(output_vocab_path):
            print(f"Found saved tokenizer config: {output_vocab_path}")
            try:
                output_vocab = HuggingFaceVocab.load(output_vocab_path)
                print(f"✓ Loaded tokenizer vocab (size: {output_vocab.vocab_size})")
            except Exception as e:
                print(f"⚠ Error loading tokenizer config: {e}. Creating new...")
                # Determine language code based on tokenizer
                if 'bartpho' in decoder_tokenizer.lower():
                    lang_code = None  # BARTPho doesn't use language codes
                else:
                    lang_code = 'vi_VN'  # mBART uses vi_VN for Vietnamese
                output_vocab = HuggingFaceVocab(decoder_tokenizer, lang_code=lang_code)
                output_vocab.save(output_vocab_path)
                print(f"Target Vocab Size (VI HuggingFace): {output_vocab.vocab_size}")
        else:
            print(f"\nCreating HuggingFace tokenizer vocab (will save to: {output_vocab_path})")
            # Determine language code based on tokenizer
            if 'bartpho' in decoder_tokenizer.lower():
                lang_code = None  # BARTPho doesn't use language codes
            else:
                lang_code = 'vi_VN'  # mBART uses vi_VN for Vietnamese
            output_vocab = HuggingFaceVocab(decoder_tokenizer, lang_code=lang_code)
            output_vocab.save(output_vocab_path)
            print(f"Target Vocab Size (VI HuggingFace): {output_vocab.vocab_size}")
    else:
        # Use traditional vocab building
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
                        print(f"❌ ERROR: Could not initialize ViWordVocab. Falling back to Word-level.")
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
                    print(f"❌ ERROR: Could not initialize ViWordVocab. Falling back to Word-level.")
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
                if isinstance(input_vocab, HuggingFaceVocab):
                    # Use HuggingFace tokenizer
                    en_indices = input_vocab.encode(en_sent, add_special_tokens=True)
                elif source_level == 'word':
                    en_indices = [input_vocab.bos_idx] + input_vocab.sentence_to_indices(en_sent) + [input_vocab.eos_idx]
                else:  # phoneme
                    en_indices = input_vocab.encode_caption(en_sent)
                
                # Target (VI)
                if isinstance(output_vocab, HuggingFaceVocab):
                    # Use HuggingFace tokenizer
                    vi_indices = output_vocab.encode(vi_sent, add_special_tokens=True)
                else:
                    # === TARGET LANGUAGE HANDLING ===
                    if vi_tok is None:
                        # dùng VI phoneme-level vocab (original)
                        vi_words = vi_sent.split()
                        vi_indices_list = vi_vocab.encode_caption(vi_words).tolist()

                    else:
                        # dùng pretrained tokenizer (mBART hoặc BARTPho)
                        ids = vi_tok.encode(vi_sent, add_special_tokens=True)
                        vi_indices_list = ids   # now it's just List[int]
                
                # Handle tensor return type for Vietnamese phonemes
                if isinstance(vi_indices, torch.Tensor):
                    vi_indices = vi_indices.tolist()
                
                # Calculate lengths for filtering
                # Handle both HuggingFace tokenizer (returns flat list) and traditional vocab
                if isinstance(input_vocab, HuggingFaceVocab):
                    en_len = len(en_indices)
                elif source_level == 'word':
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
                
                if isinstance(output_vocab, HuggingFaceVocab):
                    vi_len = len(vi_indices)
                elif target_level == 'word':
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