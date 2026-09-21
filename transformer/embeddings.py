import torch
import torch.nn as nn
import math
import numpy as np


class WordEmbedding(nn.Module):

    def __init__(self, vocab_size, embedding_dim=512):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dim)

    def forward(self, token_ids):
        embeddings = self.embedding(token_ids)
        return embeddings

