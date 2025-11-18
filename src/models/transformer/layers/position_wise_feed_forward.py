"""
Position-wise Feed-Forward Network
Based on "Attention is All You Need" (Vaswani et al., 2017)
"""

import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    Two linear transformations with ReLU activation in between.
    Applied to each position separately and identically.
    """

    def __init__(self, d_model: int, hidden: int, drop_prob: float = 0.1):
        """
        Initialize Position-wise Feed-Forward Network.
        
        Args:
            d_model: Model dimension
            hidden: Hidden dimension (typically 4 * d_model)
            drop_prob: Dropout probability
        """
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feed-forward network.
        
        Args:
            x: Input tensor
                Shape: (batch_size, length, d_model)
        
        Returns:
            output: Output tensor
                Shape: (batch_size, length, d_model)
        """
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x