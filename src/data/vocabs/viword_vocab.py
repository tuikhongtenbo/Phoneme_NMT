import os
import json
import torch
from typing import List

from .utils import preprocess_sentence
from .Vietnamese_utils import analyze_Vietnamese, compose_word
from typing import *

class ViWordVocab:
    def __init__(self, config):
        self.tokenizer = config.TOKENIZER

        self.initialize_special_tokens(config)
        
        phonemes = self.make_vocab(config.JSON_PATH)
        phonemes = list(phonemes)
        self.itos = {
            i: tok for i, tok in enumerate(self.specials + phonemes)
        }

        self.stoi = {
            tok: i for i, tok in enumerate(self.specials + phonemes)
        }

        self.specials = [self.padding_token]

    @property
    def vocab_size(self):
        """Return vocabulary size for compatibility with other vocab interfaces."""
        return len(self.itos)

    def initialize_special_tokens(self, config) -> None:
        self.padding_token = config.PAD_TOKEN
        self.bos_token = config.BOS_TOKEN
        self.eos_token = config.EOS_TOKEN
        self.unk_token = config.UNK_TOKEN
        
        self.specials = [self.padding_token, self.bos_token, self.eos_token, self.unk_token]

        self.padding_idx = 0
        self.bos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3

    def make_vocab(self, config):
        json_paths = [config.TRAIN, config.DEV, config.TEST]
        phonemes = set()

        # Collect token stats from each JSON
        for path in json_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"JSON path not found: {path}")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for key in data:
                item = data[key]
                # Handle both formats: {"key": {"caption": "..."}} and {"key": "..."}
                if isinstance(item, dict) and "caption" in item:
                    caption = item["caption"]
                elif isinstance(item, str):
                    caption = item
                else:
                    # Skip if item is neither a dict with caption nor a string
                    continue
                    
                words = preprocess_sentence(caption)
                for word in words:
                    components = analyze_Vietnamese(word)
                    if components:
                        phonemes.update([phoneme for phoneme in components if phoneme])

        return phonemes

    def encode_caption(self, caption: List[str]) -> torch.Tensor:
        syllables = [
            (self.bos_idx, self.padding_idx, self.padding_idx, self.padding_idx)
        ]
        for word in caption:
            components = analyze_Vietnamese(word)
            if components:
                syllables.append([
                    self.stoi[phoneme] if phoneme else self.padding_idx for phoneme in components
                ])
            else:
                syllables.append(
                    (self.unk_idx, self.padding_idx, self.padding_idx, self.padding_idx)
                )

        syllables.append(
            (self.eos_idx, self.padding_idx, self.padding_idx, self.padding_idx)
        )

        vec = torch.tensor(syllables).long()

        return vec

    def decode_caption(self, caption_vec: torch.Tensor, join_words=True):
        assert caption_vec.dim() == 2
        syllable_ids = caption_vec.tolist()
        syllables = [
            (
                self.itos[phoneme_idx] for phoneme_idx in phoneme_ids
            ) for phoneme_ids in syllable_ids
        ]
        sentence = []
        for phonemes in syllables:
            onset, medial, nucleus, coda = phonemes
            # turn phoneme into None if they are in special tokens
            onset = None if onset in self.specials else onset
            medial = None if medial in self.specials else medial
            nucleus = None if nucleus in self.specials else nucleus
            coda = None if coda in self.specials else coda
            word = compose_word(onset, medial, nucleus, coda)
            if word:
                sentence.append(word)
            else:
                if onset == self.bos_token:
                    sentence.append(self.bos_token)
                    continue
                
                if onset == self.eos_token:
                    sentence.append(self.eos_token)
                    continue
                
                sentence.append(self.unk_token)

        # remove the <bos> and <eos> token
        if sentence[0] == self.bos_token:
            sentence = sentence[1:]
        if sentence[-1] == self.eos_token:
            sentence = sentence[:-1]

        if join_words:
            return " ".join(sentence)
        else:
            return sentence

    def decode_batch_caption(self, caption_batch: torch.Tensor, join_words=True):
        assert caption_batch.dim() == 3
        captions = [
            self.decode_caption(caption_vec, join_words) for caption_vec in caption_batch
        ]

        return captions


if __name__ == "__main__":
    
    print("=" * 60)
    print("Testing Vietnamese Phoneme Analysis")
    print("=" * 60)
    
    # Test sentences
    test_sentences = [
        "Xin chào thế giới",
        "Tôi yêu Việt Nam",
        "Học máy thật sự là thú vị"
    ]
    
    for sentence in test_sentences:
        print(f"\n📝 Original: {sentence}")
        
        # Preprocess
        words = preprocess_sentence(sentence)
        print(f"   Preprocessed: {words}")
        
        # Analyze each word
        print(f"   Phoneme breakdown:")
        for word in words:
            components = analyze_Vietnamese(word)
            if components:
                onset, medial, nucleus, coda = components
                print(f"      '{word}' -> onset:{onset}, medial:{medial}, nucleus:{nucleus}, coda:{coda}")
                
                # Reconstruct word
                reconstructed = compose_word(onset, medial, nucleus, coda)
                match_symbol = "✓" if reconstructed == word else "✗"
                print(f"         Reconstructed: '{reconstructed}' {match_symbol}")
            else:
                print(f"      '{word}' -> Cannot analyze")
    
    print("\n" + "=" * 60)
    print("Testing with ViWordVocab (requires config)")
    print("=" * 60)
    
    # Test with ViWordVocab if config is available
    try:
        # Create a simple config object for testing
        class SimpleConfig:
            PAD_TOKEN = "<pad>"
            BOS_TOKEN = "<bos>"
            EOS_TOKEN = "<eos>"
            UNK_TOKEN = "<unk>"
            TOKENIZER = "word"
            
            class JSON_PATH:
                TRAIN = "dataset/vocabs/full_vocab_ipa.json"
                DEV = "dataset/vocabs/full_vocab_ipa.json"
                TEST = "dataset/vocabs/full_vocab_ipa.json"
        
        # Check if JSON file exists
        if os.path.exists("dataset/vocabs/full_vocab_ipa.json"):
            print("\n✓ Found vocabulary file, initializing ViWordVocab...")
            
            config = SimpleConfig()
            vocab = ViWordVocab(config)
            
            print(f"   Vocabulary size: {len(vocab.stoi)}")
            print(f"   Special tokens: {config.PAD_TOKEN}, {config.BOS_TOKEN}, {config.EOS_TOKEN}, {config.UNK_TOKEN}")
            
            # Test encode/decode
            test_caption = preprocess_sentence("Xin chào Việt Nam")
            print(f"\n📝 Test caption: {test_caption}")
            
            encoded = vocab.encode_caption(test_caption)
            print(f"   Encoded shape: {encoded.shape}")
            print(f"   Encoded tensor:\n{encoded}")
            
            decoded = vocab.decode_caption(encoded)
            print(f"   Decoded: {decoded}")
            
        else:
            print("\n⚠ Vocabulary JSON file not found.")
            print("   Please ensure 'dataset/vocabs/full_vocab_ipa.json' exists.")
            print("   Skipping ViWordVocab test.")
            
    except Exception as e:
        print(f"\n⚠ Could not test ViWordVocab: {e}")
        print("   This is expected if running without full dataset setup.")
    
    print("\n" + "=" * 60)
