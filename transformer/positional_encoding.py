import torch
import torch.nn as nn
import math
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim=512, max_seq_len=5000):
        super().__init__()

        # Create matrix
        pe = torch.zeros(max_seq_len, embedding_dim)

        # Position: 0, 1, 2, 3, ...
        position = torch.arange(0,max_seq_len,dtype=torch.float).unsqueeze(1)

        # Division term
        div_term = torch.exp(torch.arange(0,embedding_dim,2).float()* (-math.log(10000.0) / embedding_dim))

        # Even dimensions -> sin
        pe[:, 0::2] = torch.sin(position * div_term)

        # Odd dimensions -> cos
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension
        pe = pe.unsqueeze(0)

        # Register as buffer
        self.register_buffer('pe', pe)


    def forward(self, x):

        # x shape:
        # (sequence_length, embedding_dim)

        seq_len = x.size(0)

        positional_encoding = self.pe[0,:seq_len,:]

        return positional_encoding