# src/data/vocabs/base_vocab.py
"""
Base vocabulary class for word-level vocabularies.
"""

from typing import List, Dict
import os
import json

from ..constants import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, PAD_ID, SOS_ID, EOS_ID, UNK_ID
from .utils import preprocess_sentence


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
            'index2word': {str(k): v for k, v in self.index2word.items()},  
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