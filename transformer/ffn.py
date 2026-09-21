import torch
import torch.nn as nn
import math
import numpy as np


class FeedForward(nn.Module):
    def __init__(self, embedding_dim=512, ff_dim=2048):
        super().__init__()

        # First projection: 512 -> 2048
        self.linear1 = nn.Linear(embedding_dim, ff_dim)

        # ReLU activation
        self.relu = nn.ReLU()

        # Second projection: 2048 -> 512
        self.linear2 = nn.Linear(ff_dim, embedding_dim)

    def forward(self, x):
        # x shape: (sequence_length, 512)

        x = self.linear1(x)   # (seq_len, 2048)
        x = self.relu(x)      # (seq_len, 2048)
        x = self.linear2(x)   # (seq_len, 512)

        return x

