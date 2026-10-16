import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from next_word_predictor import NextWordPredictor


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"

TEST_FILE = DATA_DIR / "test_tokens.npy"
TOKENIZER_FILE = DATA_DIR / "tokenizer.json"
CHECKPOINT_FILE = CHECKPOINT_DIR / "best_model.pt"


# --------------------------------------------------
# 2. CONFIGURATION
# --------------------------------------------------

BATCH_SIZE = 16
SEQUENCE_LENGTH = 128
PAD_TOKEN_ID = 0


# --------------------------------------------------
# 3. DEVICE
# --------------------------------------------------

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Device: {DEVICE}")


# --------------------------------------------------
# 4. DATASET
# --------------------------------------------------

class LanguageModelDataset(Dataset):
    def __init__(self, tokens, sequence_length):
        self.tokens = torch.as_tensor(
            tokens,
            dtype=torch.long
        )

        self.sequence_length = sequence_length

        self.num_samples = (
            len(self.tokens) - 1
        ) // sequence_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        start = index * self.sequence_length
        end = start + self.sequence_length + 1

        chunk = self.tokens[start:end]

        inputs = chunk[:-1]
        targets = chunk[1:]

        return inputs, targets


# --------------------------------------------------
# 5. LOAD TEST DATA
# --------------------------------------------------

if not TEST_FILE.exists():
    raise FileNotFoundError(
        f"Test dataset not found: {TEST_FILE}"
    )

test_tokens = np.load(TEST_FILE).astype(np.int64)

print(f"Test tokens: {len(test_tokens):,}")

test_dataset = LanguageModelDataset(
    test_tokens,
    SEQUENCE_LENGTH
)

if len(test_dataset) == 0:
    raise ValueError(
        "Test dataset is too small for the selected sequence length."
    )

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"Test sequences: {len(test_dataset)}")


# --------------------------------------------------
# 6. LOAD MODEL CONFIGURATION
# --------------------------------------------------

if not TOKENIZER_FILE.exists():
    raise FileNotFoundError(
        f"Tokenizer configuration not found: {TOKENIZER_FILE}"
    )

with open(TOKENIZER_FILE, "r", encoding="utf-8") as file:
    tokenizer_config = json.load(file)

vocab_size = tokenizer_config["vocab_size"]

print(f"Vocabulary size: {vocab_size}")


# --------------------------------------------------
# 7. LOAD CHECKPOINT
# --------------------------------------------------

if not CHECKPOINT_FILE.exists():
    raise FileNotFoundError(
        f"Best checkpoint not found: {CHECKPOINT_FILE}"
    )

checkpoint = torch.load(
    CHECKPOINT_FILE,
    map_location=DEVICE,
    weights_only=True
)

# Support checkpoints containing either model weights
# directly or a dictionary containing model_state_dict.
if "model_state_dict" in checkpoint:
    model_state_dict = checkpoint["model_state_dict"]
else:
    model_state_dict = checkpoint

# These values must match the configuration used in train.py.
model = NextWordPredictor(
    vocab_size=vocab_size,
    embedding_dim=128,
    num_heads=4,
    ff_dim=512,
    num_layers=2,
    max_seq_len=SEQUENCE_LENGTH,
    pad_token_id=PAD_TOKEN_ID
).to(DEVICE)

model.load_state_dict(model_state_dict)

model.eval()

print("Best model checkpoint loaded successfully.")


# --------------------------------------------------
# 8. EVALUATE
# --------------------------------------------------

criterion = nn.CrossEntropyLoss(
    ignore_index=PAD_TOKEN_ID,
    reduction="sum"
)

total_loss = 0.0
total_tokens = 0

with torch.inference_mode():
    for inputs, targets in test_loader:
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        logits = model(inputs)

        # logits:  (batch_size, sequence_length, vocab_size)
        # targets: (batch_size, sequence_length)

        loss = criterion(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1)
        )

        valid_tokens = targets.ne(PAD_TOKEN_ID).sum().item()

        total_loss += loss.item()
        total_tokens += valid_tokens

        if total_tokens > 0:
            current_loss = total_loss / total_tokens
            print(
                f"\rEvaluated tokens: {total_tokens:,} | "
                f"Current loss: {current_loss:.4f}",
                end=""
            )

if total_tokens == 0:
    raise ValueError("No valid target tokens found in the test dataset.")

test_loss = total_loss / total_tokens
perplexity = math.exp(test_loss)

print("\n\n" + "=" * 45)
print("          TEST EVALUATION RESULTS")
print("=" * 45)

print(f"Test loss:       {test_loss:.4f}")
print(f"Test perplexity: {perplexity:.2f}")
print(f"Evaluated tokens: {total_tokens:,}")

print("=" * 45)


# --------------------------------------------------
# 9. SAVE RESULTS
# --------------------------------------------------

results = {
    "test_loss": test_loss,
    "test_perplexity": perplexity,
    "evaluated_tokens": total_tokens,
    "vocabulary_size": vocab_size,
    "sequence_length": SEQUENCE_LENGTH
}

results_file = CHECKPOINT_DIR / "evaluation_results.json"

with open(results_file, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print(f"\nResults saved to: {results_file}")