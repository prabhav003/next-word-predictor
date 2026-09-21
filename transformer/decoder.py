import torch
import torch.nn as nn
import math
import numpy as np

from positional_encoding import PositionalEncoding
from masked_multihead_attention import MaskedMultiHeadAttention
from cross_multihead_attention import CrossMultiHeadAttention
from add_norm import AddAndNormalize
from ffn import FeedForward


class Decoder(nn.Module):

    def __init__(
        self,
        num_layers=16,
        embedding_dim=512,
        num_heads=16,
        ff_dim=2048
    ):
        super().__init__()

        self.positional_encoding = PositionalEncoding(
            embedding_dim=embedding_dim
        )

        self.masked_multi_head_attention = MaskedMultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads
        )

        self.cross_multi_head_attention = CrossMultiHeadAttention(
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

    def forward(self, embeddings, encoder_output):

        positional_encoding = self.positional_encoding(
            embeddings
        )

        x = embeddings + positional_encoding

        for _ in range(self.num_layers):

            # Masked self-attention

            masked_attention_output = self.masked_multi_head_attention(
                x
            )

            masked_attention_output = self.add_norm(
                x,
                masked_attention_output
            )

            # Cross-attention

            cross_attention_output = self.cross_multi_head_attention(
                masked_attention_output,
                encoder_output
            )

            cross_attention_output = self.add_norm(
                masked_attention_output,
                cross_attention_output
            )

            # Feed-forward network

            feed_forward_output = self.feed_forward(
                cross_attention_output
            )

            x = self.add_norm(
                cross_attention_output,
                feed_forward_output
            )

        return x