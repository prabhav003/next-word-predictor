import json
from pathlib import Path

import numpy as np
import torch

from next_word_predictor import NextWordPredictor


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"

TOKENIZER_FILE = DATA_DIR / "tokenizer.json"
CHECKPOINT_FILE = CHECKPOINT_DIR / "best_model.pt"


# --------------------------------------------------
# 2. GENERATION CONFIGURATION
# --------------------------------------------------

EMBEDDING_DIM = 128
NUM_HEADS = 4
FF_DIM = 512
NUM_LAYERS = 2
MAX_SEQ_LEN = 128
PAD_TOKEN_ID = 0
EOS_TOKEN_ID = 2

# Generation defaults
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.8
TOP_K = 40


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
# 4. LOAD TOKENIZER
# --------------------------------------------------

if not TOKENIZER_FILE.exists():
    raise FileNotFoundError(
        f"Tokenizer configuration not found: {TOKENIZER_FILE}"
    )

with open(TOKENIZER_FILE, "r", encoding="utf-8") as file:
    tokenizer_config = json.load(file)


# --------------------------------------------------
# 5. TOKENIZER
# --------------------------------------------------

class ByteLevelBPE:
    """
    Byte-level BPE tokenizer compatible with the tokenizer
    described in this project's tokenizer.json.

    If you already have a tokenizer class in your project,
    prefer importing and using that class instead.
    """

    def __init__(self, config):
        self.vocab_size = config["vocab_size"]

        self.byte_to_id = {
            bytes([i]): i + 3
            for i in range(256)
        }

        self.id_to_byte = {
            i + 3: bytes([i])
            for i in range(256)
        }

        self.merges = []

        next_id = 259

        for merge in config["merges"]:
            left = bytes.fromhex(merge[0])
            right = bytes.fromhex(merge[1])
            merged = left + right

            self.merges.append((left, right, merged, next_id))
            self.id_to_byte[next_id] = merged

            next_id += 1

        if next_id != self.vocab_size:
            raise ValueError(
                "Tokenizer vocabulary size does not match "
                "the loaded merge rules. Use your original "
                "tokenizer implementation."
            )

    def encode(self, text):
        tokens = [bytes([b]) for b in text.encode("utf-8")]

        for left, right, merged, token_id in self.merges:
            new_tokens = []
            i = 0

            while i < len(tokens):
                if (
                    i + 1 < len(tokens)
                    and tokens[i] == left
                    and tokens[i + 1] == right
                ):
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        result = []

        for token in tokens:
            token_id = self.id_to_token_id(token)
            result.append(token_id)

        return result

    def id_to_token_id(self, token):
        # Base byte tokens.
        if len(token) == 1:
            return token[0] + 3

        # Merged tokens.
        for token_id, token_bytes in self.id_to_byte.items():
            if token_bytes == token:
                return token_id

        raise ValueError(f"Token not found in vocabulary: {token!r}")

    def decode(self, token_ids):
        output = bytearray()

        for token_id in token_ids:
            token_id = int(token_id)

            # Skip special tokens.
            if token_id in (0, 1, 2):
                continue

            if token_id not in self.id_to_byte:
                continue

            output.extend(self.id_to_byte[token_id])

        return output.decode("utf-8", errors="replace")


tokenizer = ByteLevelBPE(tokenizer_config)


# --------------------------------------------------
# 6. LOAD MODEL
# --------------------------------------------------

if not CHECKPOINT_FILE.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_FILE}"
    )

checkpoint = torch.load(
    CHECKPOINT_FILE,
    map_location=DEVICE,
    weights_only=True
)

if "model_state_dict" in checkpoint:
    model_state_dict = checkpoint["model_state_dict"]
else:
    model_state_dict = checkpoint

model = NextWordPredictor(
    vocab_size=tokenizer.vocab_size,
    embedding_dim=EMBEDDING_DIM,
    num_heads=NUM_HEADS,
    ff_dim=FF_DIM,
    num_layers=NUM_LAYERS,
    max_seq_len=MAX_SEQ_LEN,
    pad_token_id=PAD_TOKEN_ID
).to(DEVICE)

model.load_state_dict(model_state_dict)
model.eval()

print("Best model checkpoint loaded successfully.")


# --------------------------------------------------
# 7. TOP-K SAMPLING
# --------------------------------------------------

def sample_next_token(logits, temperature=0.8, top_k=40):
    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")

    logits = logits / temperature

    # Prevent selection of special tokens.
    logits[PAD_TOKEN_ID] = float("-inf")
    logits[1] = float("-inf")  # <UNK>

    # Greedy decoding.
    if top_k <= 1:
        return torch.argmax(logits).item()

    # Keep only the top-k logits.
    k = min(top_k, logits.numel())
    top_values, top_indices = torch.topk(logits, k)

    filtered_logits = torch.full_like(
        logits,
        float("-inf")
    )
    filtered_logits[top_indices] = top_values

    probabilities = torch.softmax(filtered_logits, dim=-1)

    next_token = torch.multinomial(probabilities, num_samples=1)

    return next_token.item()


# --------------------------------------------------
# 8. TEXT GENERATION
# --------------------------------------------------

@torch.inference_mode()
def generate_text(
    prompt,
    max_new_tokens=200,
    temperature=0.8,
    top_k=40
):
    token_ids = tokenizer.encode(prompt)

    if not token_ids:
        raise ValueError("Prompt must contain at least one token.")

    generated_ids = token_ids.copy()

    for _ in range(max_new_tokens):
        # Keep the most recent context tokens.
        context_ids = generated_ids[-MAX_SEQ_LEN:]

        input_tensor = torch.tensor(
            context_ids,
            dtype=torch.long,
            device=DEVICE
        )

        logits = model(input_tensor)

        # Logits for the final position predict the next token.
        next_token_logits = logits[-1].clone()

        next_token_id = sample_next_token(
            next_token_logits,
            temperature=temperature,
            top_k=top_k
        )

        # Stop if the model predicts EOS.
        if next_token_id == EOS_TOKEN_ID:
            break

        generated_ids.append(next_token_id)

    return tokenizer.decode(generated_ids)


# --------------------------------------------------
# 9. INTERACTIVE GENERATION
# --------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("       SMALL LANGUAGE MODEL")
    print("=" * 50)

    print("Enter a prompt to generate text.")
    print("Type 'exit' to quit.")

    while True:
        prompt = input("\nPrompt: ").strip()

        if prompt.lower() == "exit":
            break

        if not prompt:
            print("Please enter a prompt.")
            continue

        try:
            generated_text = generate_text(
                prompt=prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_k=TOP_K
            )

            print("\nGenerated text:")
            print("-" * 50)
            print(generated_text)
            print("-" * 50)

        except Exception as error:
            print(f"Generation failed: {error}")