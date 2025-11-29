from typing import List, Dict, Tuple, Any, Optional
import os
import torch
from collections import Counter
from transformers import AutoTokenizer
from ..vocabs.utils import preprocess_sentence
from ..vocabs.viword_vocab import ViWordVocab 
from ..vocabs.Vietnamese_util import analyze_Vietnamese 

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

def prepare_data(data_root: str, splits: List[str], max_len: int, min_count: int = 3, config: Any = None) -> Dict[str, Any]:
    """
    Loads raw data, builds vocabularies (EN word-level, VI phoneme-level), 
    converts sentences to indices, and filters based on max_len.
    """
    
    # 1. Load all raw data
    # Load pretrained tokenizers if required
    en_tok, vi_tok = load_pretrained_tokenizers(config)

    all_raw_pairs = {}
    for split in splits:
        all_raw_pairs[split] = load_pairs(data_root, split)

    if 'train' not in all_raw_pairs:
        raise ValueError("Missing 'train' split required for vocabulary building.")

    # 2. Build English Vocab (Word-level)
    # === ENGLISH TOKENIZATION HANDLING ===
    if en_tok is None:
        # --- OLD MODE: use vocab word-level ---
        input_vocab = EnWordVocab("en")

        for en_sent, _ in all_raw_pairs['train']:
            input_vocab.add_sentence(en_sent)

        input_vocab.trim(min_count)
        print(f"Input Vocab Size (EN word-level): {input_vocab.count}")

    else:
        # --- PRETRAINED MODE (mBART tokenizer) ---
        print("Using pretrained tokenizer for EN (mBART)")
        input_vocab = en_tok

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
            
            # === TARGET LANGUAGE HANDLING ===
            if vi_tok is None:
                # use VI phoneme-level vocab (original)
                vi_words = vi_sent.split()
                vi_indices_list = vi_vocab.encode_caption(vi_words).tolist()

            else:
                # use pretrained tokenizer (mBART or BARTPho)
                ids = vi_tok.encode(vi_sent, add_special_tokens=True)
                vi_indices_list = ids   # now it's just List[int]

            
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