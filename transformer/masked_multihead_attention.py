import math
import torch
import torch.nn as nn


class MaskedMultiHeadAttention(nn.Module):

    def __init__(
        self,
        embedding_dim=256,
        num_heads=8
    ):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads."
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.W_Q = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.W_K = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.W_V = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.W_O = nn.Linear(
            embedding_dim,
            embedding_dim
        )

    def forward(
        self,
        input_features,
        padding_mask=None
    ):

        # Accept either (seq_len, embedding_dim)
        # or (batch_size, seq_len, embedding_dim).

        if input_features.ndim == 2:
            input_features = input_features.unsqueeze(0)
            squeeze_batch = True

        elif input_features.ndim == 3:
            squeeze_batch = False

        else:
            raise ValueError(
                "Input must be 2D or 3D."
            )

        batch_size, seq_len, _ = input_features.shape

        # 1. Generate queries, keys and values.

        Q = self.W_Q(input_features)
        K = self.W_K(input_features)
        V = self.W_V(input_features)

        # 2. Separate attention heads.

        Q = Q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        K = K.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        V = V.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        # Shapes:
        # Q, K, V: (batch_size, num_heads, seq_len, head_dim)

        # 3. Calculate attention scores.

        scores = (
            Q @ K.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        # Shape:
        # (batch_size, num_heads, seq_len, seq_len)

        # 4. Create causal mask.
        # True means that the position is forbidden.

        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=input_features.device
            ),
            diagonal=1
        )

        scores = scores.masked_fill(
            causal_mask[None, None, :, :],
            float("-inf")
        )

        # 5. Mask padding tokens as attention keys.

        if padding_mask is not None:

            if padding_mask.ndim == 1:
                padding_mask = padding_mask.unsqueeze(0)

            if padding_mask.shape != (
                batch_size,
                seq_len
            ):
                raise ValueError(
                    "padding_mask must have shape "
                    "(batch_size, seq_len)."
                )

            padding_mask = padding_mask.to(
                device=scores.device,
                dtype=torch.bool
            )

            scores = scores.masked_fill(
                padding_mask[:, None, None, :],
                float("-inf")
            )

        # 6. Convert scores into probabilities.

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        # Prevent NaNs when a query has no valid keys,
        # which can occur for leading padding positions.

        attention_weights = torch.nan_to_num(
            attention_weights,
            nan=0.0
        )

        # 7. Apply attention weights to values.

        output = attention_weights @ V

        # 8. Combine attention heads.

        output = output.transpose(
            1,
            2
        ).contiguous().view(
            batch_size,
            seq_len,
            self.embedding_dim
        )

        # 9. Final linear projection.

        output = self.W_O(output)

        if squeeze_batch:
            output = output.squeeze(0)

        return output