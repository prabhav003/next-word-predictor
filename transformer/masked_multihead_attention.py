import torch
import torch.nn as nn
import math
import numpy as np



'''

The transformer decoder is autoregressive at inference time and non-autoregressive at training time.

masked multi head attention 

because of teacher forcing we can use the training data output no need for autoregressive modelling.

autoregressive : slow 

non autoregressive : data leakage


# attention :

embeddings --> K , Q , V --> softmax(K.Q(T) / sqrt(dk)) --> W

W = [[a , b , c],
     [d, e , f],
     [g , h , i]]


# masked attention:

mask = [[a , -inf , -inf],
       [d, e , -inf],
       [g , h , i]]

embeddings --> K , Q , V --> softmax( (K.Q(T) / sqrt(dk)) + mask() )--> W

W = [[a , 0 , 0],
     [d, e , 0],
     [g , h , i]]

'''





class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim=512, num_heads=16):
        super().__init__()

        self.num_heads = num_heads
        self.embedding_dim = embedding_dim

        # dimension of each head
        self.head_dim = embedding_dim // num_heads

        # Q, K, V
        self.W_Q = nn.Linear(embedding_dim, embedding_dim)
        self.W_K = nn.Linear(embedding_dim, embedding_dim)
        self.W_V = nn.Linear(embedding_dim, embedding_dim)

        # final linear layer
        self.W_O = nn.Linear(embedding_dim, embedding_dim)


    def forward(self, input_features):

        Q = self.W_Q(input_features)
        K = self.W_K(input_features)
        V = self.W_V(input_features)

        seq_len = input_features.shape[0]

        Q = Q.reshape(seq_len, self.num_heads, self.head_dim)
        K = K.reshape(seq_len, self.num_heads, self.head_dim)
        V = V.reshape(seq_len, self.num_heads, self.head_dim)

        heads = []

        for head in range(self.num_heads):

            q = Q[:, head, :]
            k = K[:, head, :]
            v = V[:, head, :]

            # QK^T
            scores = q @ k.T

            # scale
            scores = scores / math.sqrt(self.head_dim)

            # =====================
            # Causal Mask
            # =====================

            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=input_features.device),
                diagonal=1
            )

            scores = scores.masked_fill(mask == 1, float('-inf'))

            # softmax
            weights = torch.softmax(scores, dim=-1)

            # weighted sum of V
            output = weights @ v

            heads.append(output)

        multi_head = torch.cat(heads, dim=-1)

        output = self.W_O(multi_head)

        return output

