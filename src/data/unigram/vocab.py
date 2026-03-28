"""
Unigram vocabulary using HuggingFace tokenizers.
"""

import os
from typing import List, Union
from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace

from ..word.vocab import BaseVocab
from ..constants import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN


class UnigramVocab(BaseVocab):
    def __init__(self, config=None, name: str = 'unigram'):
        super().__init__(name)
        self.tokenizer = None
        self.config = config
        if config is not None:
            self._train(config)

    def _train(self, config):
        is_source = 'en' in self.name.lower()
        train_file = config.data.train_src if is_source else config.data.train_tgt
        vocab_size = getattr(config.data, 'vocab_size', 16000)

        self.tokenizer = Tokenizer(Unigram())
        self.tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer.train(
            [train_file],
            UnigramTrainer(vocab_size=vocab_size, special_tokens=[PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN], unk_token=UNK_TOKEN)
        )
        self._update_ids()
        print(f"[OK] Trained Unigram tokenizer for {self.name} (vocab_size: {self.tokenizer.get_vocab_size()})")

    def _update_ids(self):
        self.padding_idx = self.tokenizer.token_to_id(PAD_TOKEN)
        self.bos_idx = self.tokenizer.token_to_id(SOS_TOKEN)
        self.eos_idx = self.tokenizer.token_to_id(EOS_TOKEN)
        self.unk_idx = self.tokenizer.token_to_id(UNK_TOKEN)

    def encode_caption(self, text: Union[str, List[str]]) -> List[int]:
        if isinstance(text, list):
            text = " ".join(text)
        return [self.bos_idx] + self.tokenizer.encode(text).ids + [self.eos_idx]

    def sentence_to_indices(self, text: Union[str, List[str]]) -> List[int]:
        if isinstance(text, list):
            text = " ".join(text)
        return self.tokenizer.encode(text).ids

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size() if self.tokenizer else 0

    def save(self, filepath: str):
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained yet")
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        self.tokenizer.save(filepath)
        print(f"[OK] Saved Unigram tokenizer to: {filepath}")

    @classmethod
    def load(cls, filepath: str, name: str = None):
        vocab = cls(config=None, name=name or 'unigram')
        vocab.tokenizer = Tokenizer.from_file(filepath)
        vocab._update_ids()
        print(f"[OK] Loaded Unigram tokenizer from: {filepath} (size: {vocab.tokenizer.get_vocab_size()})")
        return vocab
