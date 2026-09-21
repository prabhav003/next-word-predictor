import torch
import torch.nn as nn
import math
import numpy as np


# =====================
# text to tokenizer
# =====================

def tokenize(text):
    text = text.lower()
    text = text.replace('?', '')
    text = text.replace("'", '')
    return text.split()

text = 'Spot. Spot saw the shiny car and said, "Wow, Kitty, your car is so bright and clean!" Kitty smiled and replied, "Thank you, Spot. I polish it every day.'

tokens = tokenize(text)

# print("Tokens:", tokens)


# =====================
# Vocabulary
# =====================

vocab = {'<UNK>': 0}

for token in tokens:
    if token not in vocab:
        vocab[token] = len(vocab)

# print("Vocabulary:", vocab)


# =====================
# Token -> Token IDs
# =====================

token_ids = [
    vocab.get(token, vocab['<UNK>'])
    for token in tokens
]

token_ids = torch.tensor(token_ids, dtype=torch.long)

# print("Token IDs:", token_ids)


# =====================
# Embedding
# =====================

class WordEmbedding(nn.Module):

    def __init__(self, vocab_size, embedding_dim=512):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dim)

    def forward(self, token_ids):
        embeddings = self.embedding(token_ids)
        return embeddings


embedding_layer = WordEmbedding(vocab_size=len(vocab),embedding_dim=512)

embeddings = embedding_layer(token_ids)

# print("Embedding shape:", embeddings.shape)


# =====================
# Positional Encoding
# =====================

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim=512, max_seq_len=5000):
        super().__init__()

        # Create matrix
        pe = torch.zeros(max_seq_len, embedding_dim)

        # Position: 0, 1, 2, 3, ...
        position = torch.arange(0,max_seq_len,dtype=torch.float).unsqueeze(1)

        # Division term
        div_term = torch.exp(torch.arange(0,embedding_dim,2).float()* (-math.log(10000.0) / embedding_dim))

        # Even dimensions -> sin
        pe[:, 0::2] = torch.sin(position * div_term)

        # Odd dimensions -> cos
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension
        pe = pe.unsqueeze(0)

        # Register as buffer
        self.register_buffer('pe', pe)


    def forward(self, x):

        # x shape:
        # (sequence_length, embedding_dim)

        seq_len = x.size(0)

        positional_encoding = self.pe[0,:seq_len,:]

        return positional_encoding


# Create positional encoding
positional_encoding = PositionalEncoding(embedding_dim=512)

pe = positional_encoding(embeddings)

# print("Positional Encoding shape:", pe.shape)


# =====================
# Input features
# =====================

input_features = embeddings + pe

# print("Final input shape:", input_features.shape)





# --------------------------
'''1.encoder : 6 Layers'''
# --------------------------

# =====================
# self attention
# =====================

# class SelfAttention(nn.Module):
#     def __init__(self, embedding_dim=512):
#         super().__init__()
#         self.embedding_dim = embedding_dim
    
#     def forward(self, input_features):
#         y_final = []
#         for i in range(len(input_features)):
#             temp = []
#             for j in range(len(input_features)):
#                 temp.append(torch.dot(input_features[i],input_features[j]))
#             w = torch.softmax(torch.stack(temp)/math.sqrt(self.embedding_dim), dim=0)
#             y = torch.zeros_like(input_features[i])
#             for j in range(len(input_features)):
#                 y = y + w[j]*input_features[j]
#             y_final.append(y)
#         return torch.stack(y_final)


# self_attention = SelfAttention(embedding_dim=512)   #define the embedding_dim accordingly

# attention = self_attention(input_features)

# print("self attention output shape:", attention.shape)

























# =====================
# multi head attention
# =====================


class MultiHeadAttention(nn.Module):
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

        Q = Q.reshape(seq_len,self.num_heads,self.head_dim) 
        K = K.reshape(seq_len,self.num_heads,self.head_dim) 
        V = V.reshape(seq_len,self.num_heads,self.head_dim) 

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

multi_attention = MultiHeadAttention(embedding_dim=512, num_heads=16)
multi_attention_output = multi_attention(input_features)
# print("multi attention output shape:", multi_attention_output.shape)





























# ===================
# add and normalize
# ===================
class AddAndNormalize(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()

        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x, sublayer_output):
        # Add residual connection
        x = x + sublayer_output

        # Normalize
        x = self.layer_norm(x)

        return x

add_norm = AddAndNormalize(embedding_dim=512)
add_norm_output = add_norm(input_features, multi_attention_output)

# print('attention output: ', add_norm_output)























# =====================
# FeedForward NN 
# =====================


class FeedForward(nn.Module):
    def __init__(self, embedding_dim=512, ff_dim=2048):
        super().__init__()

        # First projection: 512 -> 2048
        self.linear1 = nn.Linear(embedding_dim, ff_dim)

        # ReLU activation
        self.relu = nn.ReLU()

        # Second projection: 2048 -> 512
        self.linear2 = nn.Linear(ff_dim, embedding_dim)

    def forward(self, x):
        # x shape: (sequence_length, 512)

        x = self.linear1(x)   # (seq_len, 2048)
        x = self.relu(x)      # (seq_len, 2048)
        x = self.linear2(x)   # (seq_len, 512)

        return x

ffn = FeedForward(embedding_dim=512, ff_dim=2048)
ffn_output = ffn(multi_attention_output)

# print("ffn output shape:", ffn_output.shape)






























# encoder 

def encoder(text):

    tokens = tokenize(text)

    vocab = {'<UNK>': 0}
    for token in tokens:
        if token not in vocab:
            vocab[token] = len(vocab)

    token_ids = [
        vocab.get(token, vocab['<UNK>'])
        for token in tokens
    ]
    token_ids = torch.tensor(token_ids, dtype=torch.long)

    embeddings = embedding_layer(token_ids)

    pe = positional_encoding(embeddings)

    input_features = embeddings + pe

    attention_output = multi_attention(input_features)
    attention_output = add_norm(input_features, attention_output)
    ffn_output = ffn(multi_attention_output)
    ffn_output = add_norm(input_features, ffn_output)

    for i in range(5):
        attention_output = multi_attention(input_features)
        attention_output = add_norm(input_features, attention_output)
        ffn_output = ffn(multi_attention_output)
        ffn_output = add_norm(input_features, ffn_output)

    return ffn_output

encoder_output = encoder('Spot. Spot saw the shiny car and said, "Wow, Kitty, your car is so bright and clean!" Kitty smiled and replied, "Thank you, Spot. I polish it every day.')

print('encoder_output.shape', encoder_output.shape)


















# decoder : 6 Layers


# =====================
# Masked Multi Head Attention
# =====================

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


# Create masked self attention
masked_attention = MaskedMultiHeadAttention(
    embedding_dim=512,
    num_heads=16
)












# =====================
# Encoder-Decoder Attention
# =====================

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


# Create cross attention
cross_attention = CrossMultiHeadAttention(
    embedding_dim=512,
    num_heads=16
)








# =====================
# Add and Normalize
# =====================

class DecoderAddAndNormalize(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()

        self.layer_norm = nn.LayerNorm(embedding_dim)


    def forward(self, x, sublayer_output):

        # Add residual connection
        x = x + sublayer_output

        # Normalize
        x = self.layer_norm(x)

        return x


decoder_add_norm = DecoderAddAndNormalize(
    embedding_dim=512
)

















# =====================
# Decoder Layer
# =====================

class DecoderLayer(nn.Module):

    def __init__( self, embedding_dim=512, num_heads=16, ff_dim=2048):

        super().__init__()

        # Masked self attention
        self.masked_attention = MaskedMultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads
        )

        # First Add and Normalize
        self.add_norm1 = DecoderAddAndNormalize(
            embedding_dim=embedding_dim
        )

        # Cross attention
        self.cross_attention = CrossMultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads
        )

        # Second Add and Normalize
        self.add_norm2 = DecoderAddAndNormalize(
            embedding_dim=embedding_dim
        )

        # FeedForward
        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim,
            ff_dim=ff_dim
        )

        # Third Add and Normalize
        self.add_norm3 = DecoderAddAndNormalize(
            embedding_dim=embedding_dim
        )


    def forward(self, decoder_input, encoder_output):

        # =====================
        # Masked Self Attention
        # =====================

        masked_attention_output = self.masked_attention(
            decoder_input
        )

        decoder_output = self.add_norm1(
            decoder_input,
            masked_attention_output
        )


        # =====================
        # Encoder-Decoder Attention
        # =====================

        cross_attention_output = self.cross_attention(
            decoder_output,
            encoder_output
        )

        decoder_output = self.add_norm2(
            decoder_output,
            cross_attention_output
        )


        # =====================
        # FeedForward
        # =====================

        ff_output = self.feed_forward(
            decoder_output
        )

        decoder_output = self.add_norm3(
            decoder_output,
            ff_output
        )

        return decoder_output


# =====================
# Decoder
# =====================

def decoder(text, encoder_output):

    tokens = tokenize(text)

    token_ids = [
        vocab.get(token, vocab['<UNK>'])
        for token in tokens
    ]

    token_ids = torch.tensor(
        token_ids,
        dtype=torch.long
    )

    # =====================
    # Embedding
    # =====================

    embeddings = embedding_layer(token_ids)

    # =====================
    # Positional Encoding
    # =====================

    pe = positional_encoding(embeddings)

    # =====================
    # Input Features
    # =====================

    decoder_input = embeddings + pe


    # =====================
    # Decoder Layer 1
    # =====================

    decoder_layer = DecoderLayer(
        embedding_dim=512,
        num_heads=16,
        ff_dim=2048
    )

    decoder_output = decoder_layer(
        decoder_input,
        encoder_output
    )


    # =====================
    # Decoder Layers 2 - 6
    # =====================

    for i in range(5):

        decoder_layer = DecoderLayer(
            embedding_dim=512,
            num_heads=16,
            ff_dim=2048
        )

        decoder_output = decoder_layer(
            decoder_output,
            encoder_output
        )

    return decoder_output


# =====================
# Run Decoder
# =====================

decoder_input_text = 'Thank you Spot'

decoder_output = decoder(
    decoder_input_text,
    encoder_output
)

print("decoder output shape:", decoder_output.shape)