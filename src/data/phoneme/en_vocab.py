"""
English phoneme-level vocabulary.
"""

import os
import json
from typing import Dict, List, Any
from tqdm import tqdm

from ..constants import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, PAD_ID, SOS_ID, EOS_ID, UNK_ID
from ..text_utils import preprocess_sentence
from .english_utils import convert_English_IPA_to_phoneme, EnglishIPA


class EnPhonemeVocab:
    def __init__(self, config):
        self.initialize_special_tokens()
        self.word_to_ipa = self._load_ipa(config)
        phonemes = self._build_vocab()
        self.itos = {i: tok for i, tok in enumerate(self.specials + sorted(phonemes))}
        self.stoi = {tok: i for i, tok in enumerate(self.specials + sorted(phonemes))}
        self.specials = [self.padding_token]

    def initialize_special_tokens(self):
        self.padding_token = PAD_TOKEN
        self.bos_token = SOS_TOKEN
        self.eos_token = EOS_TOKEN
        self.unk_token = UNK_TOKEN
        self.specials = [self.padding_token, self.bos_token, self.eos_token, self.unk_token]
        self.padding_idx = PAD_ID
        self.bos_idx = SOS_ID
        self.eos_idx = EOS_ID
        self.unk_idx = UNK_ID

    def _load_ipa(self, config) -> Dict[str, str]:
        json_path = None
        if hasattr(config, 'data'):
            json_path = getattr(config.data, 'vocab_json_train', None)
        if not json_path or not os.path.exists(json_path):
            for path in ['dataset/vocabs/clean/full_vocab.json']:
                if os.path.exists(path):
                    json_path = path
                    break
        if not json_path or not os.path.exists(json_path):
            print("[WARN] English IPA vocabulary JSON not found")
            return {}
        print(f"Loading English IPA mapping from: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        word_to_ipa = {}
        for key, value in tqdm(data.items(), desc="Loading IPA mapping"):
            if isinstance(value, str):
                word_to_ipa[key.lower()] = value
            elif isinstance(value, dict) and 'caption' in value:
                word_to_ipa[key.lower()] = value['caption']
        print(f"[OK] Loaded {len(word_to_ipa)} word -> IPA mappings")
        return word_to_ipa

    def _build_vocab(self) -> set:
        phonemes = set(EnglishIPA.Vowels) | set(EnglishIPA.Consonants)
        for ipa_str in tqdm(self.word_to_ipa.values(), desc="Building phoneme vocab"):
            for seq in convert_English_IPA_to_phoneme(ipa_str):
                if isinstance(seq, tuple) and len(seq) == 3:
                    for p in seq:
                        if p and isinstance(p, str):
                            phonemes.add(p)
        print(f"[OK] Built phoneme vocabulary with {len(phonemes)} unique phonemes")
        return phonemes

    def encode_caption(self, sentence: str) -> List[List[int]]:
        words = preprocess_sentence(sentence)
        encoded = [[self.bos_idx]]
        for word in words:
            ipa_str = self.word_to_ipa.get(word.lower())
            if ipa_str:
                for seq in convert_English_IPA_to_phoneme(ipa_str):
                    if isinstance(seq, tuple) and len(seq) == 3:
                        for p in seq:
                            if p and isinstance(p, str):
                                encoded.append([self.stoi.get(p, self.unk_idx)])
            else:
                encoded.append([self.unk_idx])
        encoded.append([self.eos_idx])
        return encoded

    def sentence_to_indices(self, sentence: str) -> List[int]:
        return [idx for seq in self.encode_caption(sentence) for idx in seq]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def save(self, filepath: str):
        vocab_data = {
            'itos': self.itos,
            'stoi': self.stoi,
            'specials': self.specials,
            'word_to_ipa': self.word_to_ipa,
            'vocab_type': 'phoneme_level',
            'vocab_size': self.vocab_size
        }
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        print(f"[OK] Saved phoneme vocabulary to: {filepath}")

    @classmethod
    def load(cls, filepath: str, config: Any = None):
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        vocab = cls.__new__(cls)
        vocab.itos = {int(k): v for k, v in vocab_data['itos'].items()}
        vocab.stoi = vocab_data['stoi']
        vocab.specials = vocab_data.get('specials', [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN])
        vocab.initialize_special_tokens()
        if 'word_to_ipa' in vocab_data and vocab_data['word_to_ipa']:
            vocab.word_to_ipa = vocab_data['word_to_ipa']
        elif config:
            vocab.word_to_ipa = cls(config).word_to_ipa
        else:
            vocab.word_to_ipa = {}
        print(f"[OK] Loaded phoneme vocabulary from: {filepath} (size: {vocab.vocab_size})")
        return vocab
