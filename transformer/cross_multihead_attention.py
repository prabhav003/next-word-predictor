import torch
import torch.nn as nn
import math
import numpy as np




class CrossMultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim=512, num_heads=16):
        super().__init__()

        self.num_heads = num_heads
        self.embedding_dim = embedding_dim

        # dimension of each head
        self.head_dim = embedding_dim // num_heads

        # Query comes from decoder
        self.W_Q = nn.Linear(embedding_dim, embedding_dim)

        # Key and Value come from encoder
        self.W_K = nn.Linear(embedding_dim, embedding_dim)
        self.W_V = nn.Linear(embedding_dim, embedding_dim)

        # final linear layer
        self.W_O = nn.Linear(embedding_dim, embedding_dim)


    def forward(self, decoder_features, encoder_output):

        # Query from decoder
        Q = self.W_Q(decoder_features)

        # Key and Value from encoder
        K = self.W_K(encoder_output)
        V = self.W_V(encoder_output)

        decoder_seq_len = decoder_features.shape[0]
        encoder_seq_len = encoder_output.shape[0]

        Q = Q.reshape(
            decoder_seq_len,
            self.num_heads,
            self.head_dim
        )

        K = K.reshape(
            encoder_seq_len,
            self.num_heads,
            self.head_dim
        )

        V = V.reshape(
            encoder_seq_len,
            self.num_heads,
            self.head_dim
        )

        heads = []

        for head in range(self.num_heads):

            q = Q[:, head, :]
            k = K[:, head, :]
            v = V[:, head, :]

            # QK^T
            scores = q @ k.T

            # scale
            scores = scores / math.sqrt(self.head_dim)

            # softmax
            weights = torch.softmax(scores, dim=-1)

            # weighted sum of V
            output = weights @ v

            heads.append(output)

        multi_head = torch.cat(heads, dim=-1)

        output = self.W_O(multi_head)

        return output

