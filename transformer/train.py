import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from next_word_predictor import NextWordPredictor


# ============================================================
# 1. PATHS
# ============================================================


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. HYPERPARAMETERS
# ============================================================

embedding_dim = 128
number_of_attention_heads = 4
number_of_decoder_layers = 2
feed_forward_dim = 512

max_sequence_length = 128
batch_size = 16

learning_rate = 3e-4
epochs = 10

pad_token_id = 0
gradient_clip = 1.0


# ============================================================
# 3. SELECT DEVICE
# ============================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Device: {device}")


# ============================================================
# 4. LOAD TOKENIZED DATASETS
# ============================================================

train_tokens = np.load(
    DATA_DIR / "train_tokens.npy"
).astype(np.int64)

validation_tokens = np.load(
    DATA_DIR / "validation_tokens.npy"
).astype(np.int64)

print(f"Training tokens: {len(train_tokens):,}")
print(f"Validation tokens: {len(validation_tokens):,}")


# ============================================================
# 5. DETERMINE VOCABULARY SIZE
# ============================================================

# The model vocabulary must match the tokenizer's vocabulary.
# tokenizer.json stores merge rules, not the complete vocabulary.
# Your tokenizer has 259 initial tokens plus learned merges.

with open(DATA_DIR / "tokenizer.json", "r", encoding="utf-8") as file:
    tokenizer_config = json.load(file)

vocab_size = 259 + len(tokenizer_config["merges"])

print(f"Vocabulary size: {vocab_size}")


# Validate token IDs before training.

if len(train_tokens) == 0 or len(validation_tokens) == 0:
    raise ValueError("Training and validation datasets must not be empty.")

if train_tokens.min() < 0 or train_tokens.max() >= vocab_size:
    raise ValueError("Training token IDs do not match the vocabulary.")

if validation_tokens.min() < 0 or validation_tokens.max() >= vocab_size:
    raise ValueError("Validation token IDs do not match the vocabulary.")


# ============================================================
# 6. CREATE INPUT-TARGET SEQUENCES
# ============================================================

class LanguageModelDataset(Dataset):

    def __init__(self, tokens, sequence_length):
        self.tokens = torch.tensor(
            tokens,
            dtype=torch.long
        )

        self.sequence_length = sequence_length

        # Each sample contains sequence_length + 1 tokens.
        self.number_of_samples = (
            len(self.tokens) - 1
        ) // sequence_length

        if self.number_of_samples < 1:
            raise ValueError(
                "Not enough tokens to create a training sequence."
            )

    def __len__(self):
        return self.number_of_samples

    def __getitem__(self, index):

        start = index * self.sequence_length

        chunk = self.tokens[
            start : start + self.sequence_length + 1
        ]

        inputs = chunk[:-1]
        targets = chunk[1:]

        return inputs, targets


train_dataset = LanguageModelDataset(
    train_tokens,
    max_sequence_length
)

validation_dataset = LanguageModelDataset(
    validation_tokens,
    max_sequence_length
)


# ============================================================
# 7. CREATE DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=False
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False
)

print(f"Training sequences: {len(train_dataset):,}")
print(f"Validation sequences: {len(validation_dataset):,}")


# ============================================================
# 8. INITIALIZE MODEL
# ============================================================

model = NextWordPredictor(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    num_heads=number_of_attention_heads,
    ff_dim=feed_forward_dim,
    num_layers=number_of_decoder_layers,
    max_seq_len=max_sequence_length,
    pad_token_id=pad_token_id
).to(device)

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print(f"Model parameters: {parameter_count:,}")


# ============================================================
# 9. LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss(
    ignore_index=pad_token_id
)


# ============================================================
# 10. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


# ============================================================
# 11-12. TRAINING AND VALIDATION
# ============================================================

best_validation_loss = float("inf")

history = {
    "train_loss": [],
    "validation_loss": [],
    "epoch_time_seconds": []
}

for epoch in range(epochs):

    epoch_start = time.time()

    # ------------------------
    # TRAINING
    # ------------------------

    model.train()

    total_train_loss = 0.0
    total_train_tokens = 0

    for inputs, targets in train_loader:

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(inputs)

        loss = criterion(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1)
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=gradient_clip
        )

        optimizer.step()

        valid_targets = targets.ne(pad_token_id).sum().item()

        total_train_loss += loss.item() * valid_targets
        total_train_tokens += valid_targets

    train_loss = (
        total_train_loss / max(total_train_tokens, 1)
    )

    # ------------------------
    # VALIDATION
    # ------------------------

    model.eval()

    total_validation_loss = 0.0
    total_validation_tokens = 0

    with torch.no_grad():

        for inputs, targets in validation_loader:

            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)

            loss = criterion(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1)
            )

            valid_targets = targets.ne(pad_token_id).sum().item()

            total_validation_loss += (
                loss.item() * valid_targets
            )

            total_validation_tokens += valid_targets

    validation_loss = (
        total_validation_loss
        / max(total_validation_tokens, 1)
    )

    epoch_time = time.time() - epoch_start

    history["train_loss"].append(train_loss)
    history["validation_loss"].append(validation_loss)
    history["epoch_time_seconds"].append(epoch_time)

    # ------------------------
    # PRINT RESULTS
    # ------------------------

    print(
        f"Epoch [{epoch + 1}/{epochs}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Validation Loss: {validation_loss:.4f} | "
        f"Time: {epoch_time:.2f}s"
    )

    # ------------------------
    # SAVE BEST CHECKPOINT
    # ------------------------

    if validation_loss < best_validation_loss:

        best_validation_loss = validation_loss

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "vocab_size": vocab_size,
                "embedding_dim": embedding_dim,
                "num_heads": number_of_attention_heads,
                "num_layers": number_of_decoder_layers,
                "ff_dim": feed_forward_dim,
                "max_seq_len": max_sequence_length,
                "pad_token_id": pad_token_id
            },
            CHECKPOINT_DIR / "best_model.pt"
        )

        print("Best model checkpoint saved.")


# ============================================================
# 13. SAVE FINAL MODEL AND TRAINING METRICS
# ============================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epochs,
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "num_heads": number_of_attention_heads,
        "num_layers": number_of_decoder_layers,
        "ff_dim": feed_forward_dim,
        "max_seq_len": max_sequence_length,
        "pad_token_id": pad_token_id
    },
    CHECKPOINT_DIR / "final_model.pt"
)

with open(
    CHECKPOINT_DIR / "training_history.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(history, file, indent=4)

print("\nTraining complete.")
print(f"Best validation loss: {best_validation_loss:.4f}")
print(f"Checkpoints saved in: {CHECKPOINT_DIR}")