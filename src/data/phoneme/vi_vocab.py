"""
Vietnamese phoneme-level vocabulary.
"""

import os
import json
import torch
from typing import List, Any
from tqdm import tqdm

from ..text_utils import preprocess_sentence
from .vietnamese_utils import analyze_Vietnamese, compose_word


class ViWordVocab:
    def __init__(self, config):
        self.tokenizer = getattr(config, 'TOKENIZER', 'word')
        self._init_special_tokens(config)
        phonemes = self._build_vocab(config)
        self.itos = {i: tok for i, tok in enumerate(self.specials + phonemes)}
        self.stoi = {tok: i for i, tok in enumerate(self.specials + phonemes)}
        self.specials = [self.padding_token]

    def _init_special_tokens(self, config):
        self.padding_token = getattr(config, 'PAD_TOKEN', '<PAD>')
        self.bos_token = getattr(config, 'BOS_TOKEN', '<SOS>')
        self.eos_token = getattr(config, 'EOS_TOKEN', '<EOS>')
        self.unk_token = getattr(config, 'UNK_TOKEN', '<UNK>')
        self.specials = [self.padding_token, self.bos_token, self.eos_token, self.unk_token]
        self.padding_idx = 0
        self.bos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3

    def _build_vocab(self, config):
        if not hasattr(config, 'data') or not hasattr(config.data, 'train_tgt'):
            print("[WARN] Config missing data.train_tgt — building empty vocab")
            return []

        paths = [config.data.train_tgt]
        if hasattr(config.data, 'dev_tgt'):
            paths.append(config.data.dev_tgt)
        if hasattr(config.data, 'test_tgt'):
            paths.append(config.data.test_tgt)

        phonemes = set()
        print("Building Vietnamese phoneme vocabulary...")
        for path in paths:
            if not os.path.exists(path):
                print(f"[WARN] Skipping non-existent file: {path}")
                continue
            print(f"Processing: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                total_lines = sum(1 for _ in f)
            with open(path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, total=total_lines, desc=os.path.basename(path)):
                    for word in preprocess_sentence(line.strip()):
                        components = analyze_Vietnamese(word)
                        if components:
                            phonemes.update(p for p in components if p)
        print(f"[OK] Built Vietnamese phoneme vocabulary with {len(phonemes)} unique phonemes")
        return list(phonemes)

    def encode_caption(self, caption: List[str]) -> torch.Tensor:
        syllables = [(self.bos_idx, self.padding_idx, self.padding_idx, self.padding_idx)]
        for word in caption:
            components = analyze_Vietnamese(word)
            if components:
                syllables.append([
                    self.stoi.get(p, self.unk_idx) if p else self.padding_idx for p in components
                ])
            else:
                syllables.append((self.unk_idx, self.padding_idx, self.padding_idx, self.padding_idx))
        syllables.append((self.eos_idx, self.padding_idx, self.padding_idx, self.padding_idx))
        # Pad to 4-element alignment so the 1D tensor is always reshapeable to (N, 4)
        flat = [idx for syl in syllables for idx in syl]
        pad_needed = (4 - len(flat) % 4) % 4
        flat.extend([self.padding_idx] * pad_needed)
        return torch.tensor(flat, dtype=torch.long)

    def decode_caption(self, caption_vec: torch.Tensor, join_words=True):
        if caption_vec.dim() == 1:
            length = caption_vec.size(0)
            pad_needed = (4 - length % 4) % 4
            if pad_needed:
                pad_tensor = torch.full((pad_needed,), self.padding_idx, dtype=torch.long, device=caption_vec.device)
                caption_vec = torch.cat([caption_vec, pad_tensor])
            caption_vec = caption_vec.view(-1, 4)

        sentence = []
        for phoneme_ids in caption_vec.tolist():
            onset, medial, nucleus, coda = (self.itos.get(i) for i in phoneme_ids)
            onset = None if onset in self.specials else onset
            medial = None if medial in self.specials else medial
            nucleus = None if nucleus in self.specials else nucleus
            coda = None if coda in self.specials else coda
            word = compose_word(onset, medial, nucleus, coda)
            if word:
                sentence.append(word)
            elif onset == self.bos_token:
                sentence.append(self.bos_token)
            elif onset == self.eos_token:
                sentence.append(self.eos_token)
            else:
                sentence.append(self.unk_token)

        if sentence and sentence[0] == self.bos_token:
            sentence = sentence[1:]
        if sentence and sentence[-1] == self.eos_token:
            sentence = sentence[:-1]
        return " ".join(sentence) if join_words else sentence

    def decode_batch_caption(self, caption_batch: torch.Tensor, join_words=True):
        return [self.decode_caption(v, join_words) for v in caption_batch]

    def save(self, filepath: str):
        vocab_data = {
            'itos': {str(k): v for k, v in self.itos.items()},
            'stoi': self.stoi,
            'specials': self.specials,
            'tokenizer': self.tokenizer,
            'vocab_type': 'vietnamese_phoneme',
            'vocab_size': self.vocab_size,
            'padding_token': self.padding_token,
            'bos_token': self.bos_token,
            'eos_token': self.eos_token,
            'unk_token': self.unk_token
        }
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        print(f"[OK] Saved Vietnamese phoneme vocabulary to: {filepath}")

    @classmethod
    def load(cls, filepath: str, config: Any = None):
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        if config is None:
            class MinimalConfig:
                PAD_TOKEN = vocab_data.get('padding_token', '<PAD>')
                BOS_TOKEN = vocab_data.get('bos_token', '<SOS>')
                EOS_TOKEN = vocab_data.get('eos_token', '<EOS>')
                UNK_TOKEN = vocab_data.get('unk_token', '<UNK>')
                TOKENIZER = vocab_data.get('tokenizer', 'word')
            config = MinimalConfig()
        vocab = cls.__new__(cls)
        vocab.tokenizer = vocab_data.get('tokenizer', 'word')
        vocab.itos = {int(k): v for k, v in vocab_data['itos'].items()}
        vocab.stoi = vocab_data['stoi']
        vocab.specials = vocab_data.get('specials', [])
        vocab._init_special_tokens(config)
        print(f"[OK] Loaded Vietnamese phoneme vocabulary from: {filepath} (size: {vocab.vocab_size})")
        return vocab

    @property
    def vocab_size(self) -> int:
        return len(self.itos)
