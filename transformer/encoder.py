import torch
import torch.nn as nn
import math
import numpy as np

from embeddings import WordEmbedding
from positional_encoding import PositionalEncoding
from multihead_attention import MultiHeadAttention
from add_norm import AddAndNormalize
from ffn import FeedForward


class Encoder(nn.Module):

    def __init__(
        self,
        num_layers=6,
        embedding_dim=512,
        num_heads=16,
        ff_dim=2048
    ):
        super().__init__()

        self.positional_encoding = PositionalEncoding(
            embedding_dim=embedding_dim
        )

        self.multi_head_attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads
        )

        self.add_norm = AddAndNormalize(
            embedding_dim=embedding_dim
        )

        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim,
            ff_dim=ff_dim
        )

        self.num_layers = num_layers
        self.embedding_dim = embedding_dim

    def forward(self, embeddings):

        positional_encoding = self.positional_encoding(
            embeddings
        )

        x = embeddings + positional_encoding

        for _ in range(self.num_layers):

            attention_output = self.multi_head_attention(
                x
            )

            attention_output = self.add_norm(
                x,
                attention_output
            )

            feed_forward_output = self.feed_forward(
                attention_output
            )

            x = self.add_norm(
                attention_output,
                feed_forward_output
            )

        return x