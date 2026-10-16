import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self,embedding_dim=256,max_seq_len=512):
        super().__init__()

        pe = torch.zeros(
            max_seq_len,
            embedding_dim
        )

        position = torch.arange(
            max_seq_len,
            dtype=torch.float32
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                embedding_dim,
                2,
                dtype=torch.float32
            ) * (-math.log(10000.0) / embedding_dim)
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term[:pe[:, 1::2].shape[1]]
        )

        # Shape: (1, max_seq_len, embedding_dim)
        self.register_buffer(
            "pe",
            pe.unsqueeze(0)
        )

    def forward(self, token_ids):

        if token_ids.ndim not in (1, 2):
            raise ValueError(
                "Input must have shape (seq_len,) "
                "or (batch_size, seq_len)."
            )

        seq_len = token_ids.shape[-1]

        if seq_len > self.pe.shape[1]:
            raise ValueError(
                f"Sequence length {seq_len} exceeds "
                f"maximum length {self.pe.shape[1]}."
            )

        # Shape: (seq_len, embedding_dim)
        return self.pe[0, :seq_len, :]