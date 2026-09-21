import torch
import torch.nn as nn
import math
import numpy as np


class AddAndNormalize(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()

        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x, sublayer_output):
        # Add residual connection
        x = x + sublayer_output

        # Normalize
        x = self.layer_norm(x)

        return x

