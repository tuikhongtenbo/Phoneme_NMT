# src/data/data_loader.py
"""
Main data preprocessing and loading functions for NMT.
Handles raw data loading, tokenization, vocabulary building, and PyTorch DataLoader creation.
"""

from typing import List, Tuple, Dict, Any, Union
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from .constants import PAD_ID
from .word.vocab import EnWordVocab, ViWordLevelVocab
from .phoneme.vocab import EnPhonemeVocab, ViWordVocab
from .bpe.vocab import BPEVocab
from .unigram.vocab import UnigramVocab
from .text_utils import preprocess_sentence
from .helpers import create_vi_vocab_config, get_vocab_filepath

# --- TARGET ID DATASET ---
class TranslationDataset(Dataset):
    """
    PyTorch Dataset for (EN_Word_IDs, VI_Target_IDs) pairs.
    Target IDs can be 1D (Word, BPE, Unigram) or 2D (Phoneme/Syllable).
    """
    def __init__(self, indexed_pairs: List[Tuple[List[int], Union[List[int], List[List[int]]]]]):
        self.pairs = indexed_pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_indices = self.pairs[idx][0]
        tgt_indices = self.pairs[idx][1]
        
        # Handle source: can be 1D or nested list
        if isinstance(src_indices, list) and len(src_indices) > 0 and isinstance(src_indices[0], list):
            src_indices = [item[0] if isinstance(item, list) and len(item) > 0 else item for item in src_indices]
        
        src_tensor = torch.tensor(src_indices, dtype=torch.long)
        
        # Target tensor can be 1D or 2D
        if isinstance(tgt_indices, list) and len(tgt_indices) > 0 and isinstance(tgt_indices[0], list):
            tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long) 
        else:
            tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long)
        
        return src_tensor, tgt_tensor

def collate_fn_factory(target_level: str):
    """
    Factory function to create a collate_fn dynamically.
    Since phoneme targets are now flattened into 1D sequences, we can use the standard padding logic for all tokenizers.
    """
    def word_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        src_batch = [item[0] for item in batch]
        tgt_batch = [item[1] for item in batch]
        
        src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=PAD_ID)
        tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_ID)
        
        return src_padded, tgt_padded
        
    return word_collate_fn

def create_data_loader(
    indexed_pairs: List[Tuple[List[int], Union[List[int], List[List[int]]]]], 
    batch_size: int, 
    shuffle: bool = True,
    target_level: str = 'phoneme'
) -> DataLoader:
    dataset = TranslationDataset(indexed_pairs)
    collate_fn = collate_fn_factory(target_level)
    
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate_fn, 
        pin_memory=True
    )
    return data_loader

# --- MAIN DATA LOADING FUNCTIONS ---

def load_pairs(split: str, config: Any) -> List[Tuple[str, str]]:
    if not config or not hasattr(config, 'data'):
        raise ValueError("Config object with data paths is required")
    
    data_config = config.data
    
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
    
    if not os.path.exists(en_path):
        raise FileNotFoundError(f"Source file not found: {en_path}")
    if not os.path.exists(vi_path):
        raise FileNotFoundError(f"Target file not found: {vi_path}")
    
    print(f"Loading {split} data:")
    print(f"  Source: {en_path}")
    print(f"  Target: {vi_path}")
    
    with open(en_path, 'r', encoding='utf-8') as f:
        en_lines = [line.strip() for line in f if line.strip()]
    
    with open(vi_path, 'r', encoding='utf-8') as f:
        vi_lines = [line.strip() for line in f if line.strip()]
        
    if len(en_lines) != len(vi_lines):
        raise ValueError(f"Sentence count mismatch between {en_path} and {vi_path}: {len(en_lines)} vs {len(vi_lines)}")

    print(f"[OK] Loaded {len(en_lines)} sentence pairs from {split} split")
    return list(zip(en_lines, vi_lines))

def prepare_data(splits: List[str], max_len: int, min_count: int = 3, config: Any = None) -> Dict[str, Any]:
    all_raw_pairs = {}
    for split in splits:
        all_raw_pairs[split] = load_pairs(split, config)

    if 'train' not in all_raw_pairs:
        raise ValueError("Missing 'train' split required for vocabulary building.")

    tokenizer_type = None
    if config and hasattr(config, 'data'):
        tokenizer_type = getattr(config.data, 'tokenizer_type', None)
        if tokenizer_type is not None:
            source_level = 'pretrained'
            target_level = 'pretrained'
            print(f"Using pretrained tokenizer type: **{tokenizer_type}**")
        else:
            source_level = getattr(config.data, 'source_level', 'word').lower()
            target_level = getattr(config.data, 'target_level', 'word').lower()
    else:
        source_level = 'word'
        target_level = 'phoneme'
    
    print(f"Source (EN) tokenization level: **{source_level}**")
    print(f"Target (VI) tokenization level: **{target_level}**")

    if tokenizer_type is not None:
        print("\n" + "="*60)
        print("Initializing pretrained tokenizers...")
        print("="*60)
        if tokenizer_type == "pretrained_1":
            source_tokenizer_name = "facebook/mbart-large-50"
            target_tokenizer_name = "facebook/mbart-large-50"
            source_tokenizer = AutoTokenizer.from_pretrained(source_tokenizer_name)
            source_tokenizer.src_lang = "en_XX"
            target_tokenizer = AutoTokenizer.from_pretrained(target_tokenizer_name)
            target_tokenizer.tgt_lang = "vi_VN"
        elif tokenizer_type == "pretrained_2":
            source_tokenizer_name = "facebook/mbart-large-50"
            target_tokenizer_name = "vinai/bartpho-word"
            source_tokenizer = AutoTokenizer.from_pretrained(source_tokenizer_name)
            source_tokenizer.src_lang = "en_XX"
            target_tokenizer = AutoTokenizer.from_pretrained(target_tokenizer_name)
        else:
            raise ValueError(f"Unknown tokenizer_type: {tokenizer_type}")
        
        class PretrainedVocabWrapper:
            def __init__(self, tokenizer, name="pretrained"):
                self.tokenizer = tokenizer
                self.name = name
                self.vocab_size = len(tokenizer)
                self.count = self.vocab_size
                self.pad_id = tokenizer.pad_token_id
                self.bos_id = tokenizer.bos_token_id
                self.eos_id = tokenizer.eos_token_id
                self.unk_id = tokenizer.unk_token_id
                self.padding_idx = self.pad_id
                self.bos_idx = self.bos_id
                self.eos_idx = self.eos_id
                self.unk_idx = self.unk_id
            
            def encode_caption(self, text: str, add_special_tokens: bool = True):
                encoded = self.tokenizer(text, add_special_tokens=add_special_tokens, return_tensors=None, padding=False, truncation=False)
                return encoded['input_ids']
            
            def sentence_to_indices(self, sentence: str):
                return self.encode_caption(sentence, add_special_tokens=False)
        
        input_vocab = PretrainedVocabWrapper(source_tokenizer, "source")
        output_vocab = PretrainedVocabWrapper(target_tokenizer, "target")
        
        print(f"[OK] Source tokenizer vocab size: {input_vocab.vocab_size}")
        print(f"[OK] Target tokenizer vocab size: {output_vocab.vocab_size}")
        print("="*60 + "\n")
        
        indexed_data = {}
        for split, pairs in all_raw_pairs.items():
            current_indexed_pairs = []
            progress_bar = tqdm(pairs, desc=f"Encoding {split}", file=sys.stdout, mininterval=1.0, ncols=100)
            
            for idx, (en_sent, vi_sent) in enumerate(progress_bar):
                try:
                    en_indices = input_vocab.encode_caption(en_sent, add_special_tokens=True)
                    vi_indices = output_vocab.encode_caption(vi_sent, add_special_tokens=True)
                    
                    if len(en_indices) <= max_len and len(vi_indices) <= max_len:
                        current_indexed_pairs.append((en_indices, vi_indices))
                    
                    if (idx + 1) % 1000 == 0:
                        progress_bar.set_postfix({'processed': f'{idx+1:,}/{len(pairs):,}', 'kept': f'{len(current_indexed_pairs):,}'})
                except Exception as e:
                    continue
            
            progress_bar.close()
            indexed_data[split] = current_indexed_pairs
        
        return {
            'input_vocab': input_vocab,
            'output_vocab': output_vocab,
            'data': indexed_data,
            'source_level': source_level,
            'target_level': target_level
        }

    input_vocab_path, output_vocab_path = get_vocab_filepath(source_level, target_level, min_count)

    if os.path.exists(input_vocab_path):
        print(f"\nFound saved input vocabulary: {input_vocab_path}")
        try:
            if source_level == 'word':
                input_vocab = EnWordVocab.load(input_vocab_path, name="en")
            elif source_level == 'phoneme':
                input_vocab = EnPhonemeVocab.load(input_vocab_path, config)
            elif source_level == 'bpe':
                input_vocab = BPEVocab.load(input_vocab_path, name="en_bpe")
            elif source_level == 'unigram':
                input_vocab = UnigramVocab.load(input_vocab_path, name="en_unigram")
            else:
                raise ValueError("Unknown source_level")
        except Exception as e:
            print(f"[WARN] Error loading vocabulary: {e}. Rebuilding...")
            if source_level == 'word':
                input_vocab = EnWordVocab("en")
                for en_sent, _ in all_raw_pairs['train']:
                    input_vocab.add_sentence(en_sent)
                input_vocab.trim(min_count)
                input_vocab.save(input_vocab_path)
            elif source_level == 'phoneme':
                input_vocab = EnPhonemeVocab(config)
                input_vocab.save(input_vocab_path)
            elif source_level == 'bpe':
                input_vocab = BPEVocab(config, name="en_bpe")
                input_vocab.save(input_vocab_path)
            elif source_level == 'unigram':
                input_vocab = UnigramVocab(config, name="en_unigram")
                input_vocab.save(input_vocab_path)
    else:
        print(f"\nBuilding input vocabulary (will save to: {input_vocab_path})")
        if source_level == 'word':
            input_vocab = EnWordVocab("en")
            for en_sent, _ in all_raw_pairs['train']:
                input_vocab.add_sentence(en_sent)
            input_vocab.trim(min_count)
            input_vocab.save(input_vocab_path)
        elif source_level == 'phoneme':
            input_vocab = EnPhonemeVocab(config)
            input_vocab.save(input_vocab_path)
        elif source_level == 'bpe':
            input_vocab = BPEVocab(config, name="en_bpe")
            input_vocab.save(input_vocab_path)
        elif source_level == 'unigram':
             input_vocab = UnigramVocab(config, name="en_unigram")
             input_vocab.save(input_vocab_path)
        else:
            raise ValueError(f"Unknown source_level: {source_level}")

    if os.path.exists(output_vocab_path):
        print(f"\nFound saved output vocabulary: {output_vocab_path}")
        try:
            if target_level == 'word':
                output_vocab = ViWordLevelVocab.load(output_vocab_path, name='vi_word')
            elif target_level == 'phoneme':
                vi_vocab_config = create_vi_vocab_config(config)
                output_vocab = ViWordVocab.load(output_vocab_path, vi_vocab_config)
            elif target_level == 'bpe':
                output_vocab = BPEVocab.load(output_vocab_path, name='vi_bpe')
            elif target_level == 'unigram':
                output_vocab = UnigramVocab.load(output_vocab_path, name='vi_unigram')
        except Exception as e:
            print(f"[WARN] Error loading vocabulary: {e}. Rebuilding...")
            if target_level == 'word':
                output_vocab = ViWordLevelVocab(config)
                for _, vi_sent in all_raw_pairs['train']:
                    output_vocab.add_sentence(vi_sent)
                output_vocab.trim(min_count)
                output_vocab.save(output_vocab_path)
            elif target_level == 'phoneme':
                vi_vocab_config = create_vi_vocab_config(config)
                output_vocab = ViWordVocab(vi_vocab_config)
                output_vocab.save(output_vocab_path)
            elif target_level == 'bpe':
                output_vocab = BPEVocab(config, name='vi_bpe')
                output_vocab.save(output_vocab_path)
            elif target_level == 'unigram':
                output_vocab = UnigramVocab(config, name='vi_unigram')
                output_vocab.save(output_vocab_path)
    else:
        print(f"\nBuilding output vocabulary (will save to: {output_vocab_path})")
        if target_level == 'word':
            output_vocab = ViWordLevelVocab(config)
            for _, vi_sent in all_raw_pairs['train']:
                output_vocab.add_sentence(vi_sent)
            output_vocab.trim(min_count)
            output_vocab.save(output_vocab_path)
        elif target_level == 'phoneme':
            vi_vocab_config = create_vi_vocab_config(config)
            output_vocab = ViWordVocab(vi_vocab_config)
            output_vocab.save(output_vocab_path)
        elif target_level == 'bpe':
            output_vocab = BPEVocab(config, name='vi_bpe')
            output_vocab.save(output_vocab_path)
        elif target_level == 'unigram':
            output_vocab = UnigramVocab(config, name='vi_unigram')
            output_vocab.save(output_vocab_path)

    print("\nConverting sentences to indices...")
    indexed_data = {}
    for split, pairs in all_raw_pairs.items():
        current_indexed_pairs = []
        progress_bar = tqdm(pairs, desc=f"Encoding {split}", file=sys.stdout, mininterval=1.0, ncols=100)
        
        for idx, (en_sent, vi_sent) in enumerate(progress_bar):
            try:
                # Source
                if source_level in ['word', 'bpe', 'unigram']:
                    # Assuming for bpe and unigram the scaffold acts similarly for now
                    en_indices = [input_vocab.bos_idx] + input_vocab.sentence_to_indices(en_sent) + [input_vocab.eos_idx]
                else:  # phoneme
                    en_indices = input_vocab.encode_caption(en_sent)
                
                # Target
                vi_words = preprocess_sentence(vi_sent)
                vi_indices = output_vocab.encode_caption(vi_words)
                
                if isinstance(vi_indices, torch.Tensor):
                    vi_indices = vi_indices.tolist()
                
                if source_level in ['word', 'bpe', 'unigram']:
                    en_len = len(en_indices)
                else: 
                    # handle phoneme logic
                    if isinstance(en_indices, list) and len(en_indices) > 0 and isinstance(en_indices[0], list):
                        en_len = sum(len(item) if isinstance(item, list) else 1 for item in en_indices)
                    else:
                        en_len = len(en_indices)
                
                if target_level in ['word', 'bpe', 'unigram', 'pretrained']:
                    vi_len = len(vi_indices)
                else:  # phoneme
                    # Phonemes are returned as a flat list, and each word generates 4 tokens + BOS/EOS (total 2 tokens for BOS/EOS)
                    if isinstance(vi_indices, torch.Tensor):
                        vi_len = vi_indices.size(0) // 4
                    elif isinstance(vi_indices, list):
                        vi_len = len(vi_indices) // 4
                    else:
                        vi_len = 0
                
                if en_len <= max_len and vi_len <= max_len:
                    current_indexed_pairs.append((en_indices, vi_indices))
                    
            except Exception as e:
                continue
        
        progress_bar.close()
        indexed_data[split] = current_indexed_pairs
    
    return {
        'input_vocab': input_vocab,
        'output_vocab': output_vocab,
        'data': indexed_data,
        'source_level': source_level,
        'target_level': target_level
    }