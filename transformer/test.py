import torch
import torch.nn as nn
import math
import numpy as np

from tokenizer import Tokenizer
from embeddings import WordEmbedding

from decoder import Decoder
from encoder import Encoder


# =========================================================
# TEXT
# =========================================================

text = '''
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.
'''


# =========================================================
# TOKENIZER
# =========================================================

tokenizer = Tokenizer()

tokens = tokenizer.tokenize(
    text
)

print("Tokens:")
print(tokens)


# =========================================================
# VOCAB
# =========================================================

vocab = {
    '<PAD>': 0,
    '<UNK>': 1,
    '<BOS>': 2,
    '<EOS>': 3
}

for token in tokens:

    if token not in vocab:
        vocab[token] = len(vocab)


print("\nVocabulary:")
print(vocab)


# =========================================================
# TOKEN IDS
# =========================================================

token_ids = torch.tensor(
    [
        vocab.get(
            token,
            vocab['<UNK>']
        )
        for token in tokens
    ],
    dtype=torch.long
)

print("\nToken IDs:")
print(token_ids)


# =========================================================
# EMBEDDINGS
# =========================================================

embedding_dim = 512

embedding_layer = WordEmbedding(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim
)

embeddings = embedding_layer(
    token_ids
)

print("\nEncoder embeddings:")
print(embeddings.shape)


# =========================================================
# ENCODER
# =========================================================

encoder_layer = Encoder(
    num_layers=6,
    embedding_dim=embedding_dim,
    num_heads=16,
    ff_dim=2048
)

encoder_output = encoder_layer(
    embeddings
)

print("\nEncoder output:")
print(encoder_output.shape)


# =========================================================
# DECODER
# =========================================================

# ---------------------------------------------------------
# Token IDs with EOS
# ---------------------------------------------------------

target_token_ids = torch.cat(
    [
        token_ids,
        torch.tensor(
            [vocab['<EOS>']],
            dtype=torch.long
        )
    ]
)

print("\nTarget token IDs:")
print(target_token_ids)


# ---------------------------------------------------------
# Right shift
# ---------------------------------------------------------

bos_token = torch.tensor(
    [vocab['<BOS>']],
    dtype=torch.long
)

decoder_token_ids = torch.cat(
    [
        bos_token,
        target_token_ids[:-1]
    ]
)

print("\nDecoder token IDs:")
print(decoder_token_ids)


# =========================================================
# DECODER EMBEDDINGS
# =========================================================

decoder_embedding_layer = WordEmbedding(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim
)

decoder_embeddings = decoder_embedding_layer(
    decoder_token_ids
)

print("\nDecoder embeddings:")
print(decoder_embeddings.shape)


# =========================================================
# DECODER
# =========================================================

decoder_layer = Decoder(
    num_layers=16,
    embedding_dim=embedding_dim,
    num_heads=16,
    ff_dim=2048
)

decoder_output = decoder_layer(
    decoder_embeddings,
    encoder_output
)

print("\nDecoder output:")
print(decoder_output.shape)

