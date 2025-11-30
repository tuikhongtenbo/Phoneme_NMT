# src/data/vocabs/en_vocabs.py
"""
English vocabulary classes for word-level and phoneme-level tokenization.
"""

from typing import List, Dict, Any
import os
import json
from tqdm import tqdm

from ..constants import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, PAD_ID, SOS_ID, EOS_ID, UNK_ID
from .base_vocab import BaseVocab
from .utils import preprocess_sentence
from .English_utils import convert_English_IPA_to_phoneme, EnglishIPA


class EnWordVocab(BaseVocab):
    """Word-level vocabulary for English (Source). Inherits from BaseVocab."""
    pass


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
                phoneme_seqs = convert_English_IPA_to_phoneme(ipa_str)
                for phoneme_seq in phoneme_seqs:
                    if isinstance(phoneme_seq, tuple) and len(phoneme_seq) == 3:
                        initial, vowel, final = phoneme_seq
                        # Add each non-empty string phoneme
                        for phoneme in [initial, vowel, final]:
                            if phoneme and isinstance(phoneme, str):
                                phonemes.add(phoneme)
        
        print(f"✓ Built phoneme vocabulary with {len(phonemes)} unique phonemes")
        return phonemes
    
    def encode_caption(self, sentence: str) -> List[List[int]]:
        """
        Convert English sentence to phoneme indices.
        
        Process: English words -> lookup IPA from JSON (via config) -> convert IPA to phonemes -> encode to indices
        """
        words = preprocess_sentence(sentence)
        encoded = [[self.bos_idx]]
        
        for word in words:
            # Lookup IPA from JSON mapping
            ipa_str = self.word_to_ipa.get(word.lower(), None)
            
            if ipa_str:
                # Convert IPA string to phonemes
                phoneme_seqs = convert_English_IPA_to_phoneme(ipa_str)
                for phoneme_seq in phoneme_seqs:
                    if isinstance(phoneme_seq, tuple) and len(phoneme_seq) == 3:
                        initial, vowel, final = phoneme_seq
                        # Encode each non-empty phoneme
                        for phoneme in [initial, vowel, final]:
                            if phoneme and isinstance(phoneme, str):
                                encoded.append([self.stoi.get(phoneme, self.unk_idx)])
            else:
                # Word not found in mapping, use UNK
                encoded.append([self.unk_idx])
        
        encoded.append([self.eos_idx])
        return encoded
    
    def sentence_to_indices(self, sentence: str) -> List[int]:
        """Convert sentence to indices."""
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
            # Create a temporary instance to use load_ipa_mapping method
            temp_vocab = cls(config)
            vocab.word_to_ipa = temp_vocab.word_to_ipa
        else:
            vocab.word_to_ipa = {}
        
        print(f"✓ Loaded phoneme vocabulary from: {filepath} (size: {vocab.vocab_size})")
        return vocab