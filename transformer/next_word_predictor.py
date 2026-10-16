import torch
import torch.nn as nn

from embeddings import WordEmbedding
from positional_encoding import PositionalEncoding
from masked_multihead_attention import MaskedMultiHeadAttention
from add_norm import AddAndNormalize
from ffn import FeedForward


class TransformerDecoderBlock(nn.Module):

    def __init__(
        self,
        embedding_dim=256,
        num_heads=8,
        ff_dim=1024
    ):
        super().__init__()

        self.masked_attention = MaskedMultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads
        )

        self.add_norm1 = AddAndNormalize(
            embedding_dim=embedding_dim
        )

        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim,
            ff_dim=ff_dim
        )

        self.add_norm2 = AddAndNormalize(
            embedding_dim=embedding_dim
        )

    def forward(
        self,
        x,
        padding_mask=None
    ):

        # 1. Masked self-attention.

        attention_output = self.masked_attention(
            x,
            padding_mask=padding_mask
        )

        # 2. Residual connection and normalization.

        x = self.add_norm1(
            x,
            attention_output
        )

        # 3. Feed-forward network.

        ff_output = self.feed_forward(x)

        # 4. Residual connection and normalization.

        x = self.add_norm2(
            x,
            ff_output
        )

        return x


class NextWordPredictor(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=256,
        num_heads=8,
        ff_dim=1024,
        num_layers=4,
        max_seq_len=512,
        pad_token_id=0
    ):
        super().__init__()

        if vocab_size <= 0:
            raise ValueError(
                "vocab_size must be positive."
            )

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads."
            )

        if num_layers <= 0 or max_seq_len <= 0:
            raise ValueError(
                "num_layers and max_seq_len must be positive."
            )

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

        # 1. Token embeddings.

        self.embedding = WordEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_token_id
        )

        # 2. Positional encoding.

        self.positional_encoding = PositionalEncoding(
            embedding_dim=embedding_dim,
            max_seq_len=max_seq_len
        )

        # 3. Transformer decoder blocks.

        self.decoder_blocks = nn.ModuleList([
            TransformerDecoderBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                ff_dim=ff_dim
            )
            for _ in range(num_layers)
        ])

        # 4. Final normalization.

        self.final_norm = nn.LayerNorm(
            embedding_dim
        )

        # 5. Predict vocabulary logits.

        self.output_layer = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, token_ids):

        if token_ids.dtype != torch.long:
            token_ids = token_ids.long()

        # Accept (seq_len,) or (batch_size, seq_len).

        single_sequence = token_ids.ndim == 1

        if single_sequence:
            token_ids = token_ids.unsqueeze(0)

        elif token_ids.ndim != 2:
            raise ValueError(
                "Input must have shape (seq_len,) "
                "or (batch_size, seq_len)."
            )

        _, seq_len = token_ids.shape

        if seq_len == 0:
            raise ValueError(
                "Input sequence cannot be empty."
            )

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds "
                f"max_seq_len={self.max_seq_len}."
            )

        if torch.any(token_ids < 0) or torch.any(
            token_ids >= self.vocab_size
        ):
            raise ValueError(
                "Token IDs are outside the vocabulary."
            )

        # True identifies padding positions.

        padding_mask = token_ids.eq(
            self.pad_token_id
        )

        # 1. Embeddings.

        x = self.embedding(token_ids)

        # 2. Add positional information.

        positional = self.positional_encoding(
            token_ids
        )

        x = x + positional.unsqueeze(0)

        # 3. Decoder blocks.

        for block in self.decoder_blocks:

            x = block(
                x,
                padding_mask=padding_mask
            )

        # 4. Final normalization.

        x = self.final_norm(x)

        # 5. Vocabulary logits.

        logits = self.output_layer(x)

        # Single input:
        # (seq_len, vocab_size)
        #
        # Batch:
        # (batch_size, seq_len, vocab_size)

        if single_sequence:
            logits = logits.squeeze(0)

        return logits

    @torch.no_grad()
    def predict_next_token(self, token_ids):

        was_training = self.training

        self.eval()

        try:

            if not torch.is_tensor(token_ids):
                token_ids = torch.tensor(
                    token_ids,
                    dtype=torch.long
                )

            device = next(
                self.parameters()
            ).device

            token_ids = token_ids.to(device)

            # Keep only the most recent context.

            if token_ids.ndim == 1:

                context = token_ids[
                    -self.max_seq_len:
                ]

                logits = self.forward(context)

                next_token_logits = logits[-1]

            elif token_ids.ndim == 2:

                context = token_ids[
                    :, -self.max_seq_len:
                ]

                logits = self.forward(context)

                next_token_logits = logits[:, -1, :]

            else:

                raise ValueError(
                    "Input must be one- or two-dimensional."
                )

            probabilities = torch.softmax(
                next_token_logits,
                dim=-1
            )

            next_token_id = probabilities.argmax(
                dim=-1
            )

            if next_token_id.numel() == 1:

                return (
                    next_token_id.item(),
                    probabilities.squeeze(0)
                    if probabilities.ndim == 2
                    else probabilities
                )

            return next_token_id, probabilities

        finally:

            self.train(was_training)