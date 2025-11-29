# src/data/data_loader.py

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Any, Union

# Import PAD_ID from preprocessing for padding
from .preprocessing import PAD_ID 

class TranslationDataset(Dataset):
    """PyTorch Dataset for (EN_Word_IDs, VI_Syllable_IDs) pairs."""
    def __init__(self, indexed_pairs: List[Tuple[List[int], List[List[int]]]]):
        self.pairs = indexed_pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # Get source and target
        src_indices = self.pairs[idx][0]  # EN: Word-level (1D)
        tgt_indices = self.pairs[idx][1]  # VI: 1D (word) or 2D (syllable/phoneme)

        # Convert source to tensor (always 1D)
        src_tensor = torch.tensor(src_indices, dtype=torch.long)

        # Target tensor: could be 1D (word-level) or 2D (syllable/phoneme level)
        if isinstance(tgt_indices, list) and len(tgt_indices) > 0 and isinstance(tgt_indices[0], list):
            # 2D target (syllable or phoneme)
            tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long)
        else:
            # 1D target
            tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long)

        return src_tensor, tgt_tensor

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom Collate function for dynamic padding.
    Source output: (Batch_size, Src_len)
    Target output: (Batch_size, Tgt_len, 4)
    """
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

def create_data_loader(
    indexed_pairs: List[Tuple[List[int], List[List[int]]]], 
    batch_size: int, 
    shuffle: bool = True
) -> DataLoader:
    """Creates a PyTorch DataLoader."""
    dataset = TranslationDataset(indexed_pairs)
    
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate_fn, 
        pin_memory=True
    )

    return data_loader
