import torch.nn as nn


class WordEmbedding(nn.Module):

    def __init__(self,vocab_size,embedding_dim=256,padding_idx=0):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )

    def forward(self, token_ids):
        return self.embedding(token_ids)