# src/data/data_loader.py

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Any, Union

# Import ID and Target Level from preprocessing
from .preprocessing import PAD_ID 

# --- Update TranslationDataset to handle both types of target indices ---
class TranslationDataset(Dataset):
    """
    PyTorch Dataset for (EN_Word_IDs, VI_Target_IDs) pairs.
    Target IDs can be 1D (Word-level) or 2D (Phoneme/Syllable-level).
    """
    def __init__(self, indexed_pairs: List[Tuple[List[int], Union[List[int], List[List[int]]]]]):
        self.pairs = indexed_pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_indices = self.pairs[idx][0] # List[int] or List[List[int]] - EN
        tgt_indices = self.pairs[idx][1] # List[int] or List[List[int]] - VI
        
        # Handle source: can be 1D (word) or nested list (phoneme)
        if isinstance(src_indices, list) and len(src_indices) > 0 and isinstance(src_indices[0], list):
            src_indices = [item[0] if isinstance(item, list) and len(item) > 0 else item for item in src_indices]
        
        src_tensor = torch.tensor(src_indices, dtype=torch.long)
        
        # Target tensor can be 1D (Word) or 2D (Syllable/Phoneme)
        if isinstance(tgt_indices, list) and len(tgt_indices) > 0 and isinstance(tgt_indices[0], list):
            # Phoneme/Syllable-level (2D)
            tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long) 
        else:
            # Word-level (1D)
            tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long)
        
        return src_tensor, tgt_tensor


def collate_fn_factory(target_level: str):
    """
    Factory function to create a collate_fn dynamically based on the target tokenization level.
    """
    
    if target_level == 'word':
        # --- Word-Level Collate Function ---
        def word_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
            """Handles 1D target tensors (Word-level)."""
            src_batch = [item[0] for item in batch]
            tgt_batch = [item[1] for item in batch] # Tensors of shape (Tgt_len,)
            
            # Pad Source (Word-level)
            src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=PAD_ID)
            
            # Pad Target (Word-level)
            # Use pad_sequence directly for 1D targets
            tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_ID)
            
            return src_padded, tgt_padded
            
        return word_collate_fn
        
    else: # Default: 'phoneme'
        # --- Phoneme-Level Collate Function ---
        def phoneme_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
            """Handles 2D target tensors (Phoneme/Syllable-level)."""
            src_batch = [item[0] for item in batch]
            tgt_batch = [item[1] for item in batch] # Tensors of shape (Tgt_len, 4)
            
            # 1. Pad Source (Word-level)
            src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=PAD_ID)
            
            # 2. Pad Target (Syllable-level - 2D Tensor Padding)
            max_tgt_len = max(t.size(0) for t in tgt_batch)
            
            tgt_padded = []
            for tgt_tensor in tgt_batch:
                current_len = tgt_tensor.size(0)
                
                if current_len < max_tgt_len:
                    # Size to pad: (max_tgt_len - current_len, 4 components)
                    pad_size = (max_tgt_len - current_len, 4)
                    
                    # Create padding tensor (PAD_ID for all 4 components)
                    padding = torch.full(pad_size, PAD_ID, dtype=torch.long)
                    padded_tensor = torch.cat([tgt_tensor, padding], dim=0)
                else:
                    padded_tensor = tgt_tensor
                    
                tgt_padded.append(padded_tensor)
                
            # Stack tensors to get (Batch_size, Max_tgt_len, 4)
            tgt_padded = torch.stack(tgt_padded, dim=0) 
            
            return src_padded, tgt_padded
            
        return phoneme_collate_fn


def create_data_loader(
    indexed_pairs: List[Tuple[List[int], Union[List[int], List[List[int]]]]], 
    batch_size: int, 
    shuffle: bool = True,
    target_level: str = 'phoneme' # Pass the determined level from prepare_data
) -> DataLoader:
    """Creates a PyTorch DataLoader, selecting collate_fn based on target_level."""
    dataset = TranslationDataset(indexed_pairs)
    
    # Select the appropriate collate function
    collate_fn = collate_fn_factory(target_level)
    
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate_fn, 
        pin_memory=True
    )
    return data_loader# Data loading utilities