import torch

from next_word_predictor import NextWordPredictor

VOCAB_SIZE = 8000

model = NextWordPredictor(
    vocab_size=VOCAB_SIZE,
    embedding_dim=512,
    num_heads=16,
    ff_dim=2048,
    num_layers=6
)

# Example token IDs
token_ids = torch.tensor([101, 428, 16, 3205])

# Forward pass
logits = model(token_ids)

print("Logits shape:", logits.shape)

# Predict the next token
next_token_id, probabilities = model.predict_next_token(token_ids)

print("Predicted token ID:", next_token_id)
print("Probability:", probabilities[next_token_id].item())