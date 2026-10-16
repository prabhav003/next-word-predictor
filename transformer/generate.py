import json
from pathlib import Path

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
# 2. MODEL CONFIGURATION
# Must match train.py
# --------------------------------------------------

EMBEDDING_DIM = 128
NUM_HEADS = 4
FF_DIM = 512
NUM_LAYERS = 2
MAX_SEQ_LEN = 128

PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1
EOS_TOKEN_ID = 2

# Generation settings
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
# 4. TOKENIZER
# Matches your original tokenizer.py
# --------------------------------------------------

class Tokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.merges = []

        self.vocab = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<EOS>": 2
        }

        self.id_to_token = {}

        self._add_byte_tokens()

    def _add_byte_tokens(self):
        for i in range(256):
            token = bytes([i])

            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

        self.id_to_token = {
            token_id: token
            for token, token_id in self.vocab.items()
        }

    def get_pairs(self, tokens):
        from collections import Counter
        return Counter(zip(tokens, tokens[1:]))

    def merge_pair(self, tokens, pair):
        merged = []
        i = 0

        while i < len(tokens):
            if (
                i + 1 < len(tokens)
                and (tokens[i], tokens[i + 1]) == pair
            ):
                merged.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                merged.append(tokens[i])
                i += 1

        return merged

    def tokenize(self, text):
        tokens = [
            bytes([b])
            for b in text.encode("utf-8")
        ]

        for pair in self.merges:
            tokens = self.merge_pair(tokens, pair)

        return tokens

    def encode(self, text):
        tokens = self.tokenize(text)

        return [
            self.vocab.get(token, self.vocab["<UNK>"])
            for token in tokens
        ]

    def encode_document(self, text):
        return self.encode(text) + [EOS_TOKEN_ID]

    def decode(self, ids):
        byte_tokens = []

        for token_id in ids:
            token_id = self.id_to_token.get(int(token_id))

            if token_id is None:
                raise ValueError(
                    f"Unknown token ID: {token_id}"
                )

            if token_id == "<PAD>":
                continue

            elif token_id == "<EOS>":
                byte_tokens.append(b"\n\n")

            elif token_id == "<UNK>":
                byte_tokens.append(
                    "\uFFFD".encode("utf-8")
                )

            elif isinstance(token_id, bytes):
                byte_tokens.append(token_id)

        return b"".join(byte_tokens).decode(
            "utf-8",
            errors="replace"
        )

    def load(self, path):
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        self.vocab_size = data["vocab_size"]

        self.merges = [
            (bytes.fromhex(a), bytes.fromhex(b))
            for a, b in data["merges"]
        ]

        self.vocab = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<EOS>": 2
        }

        self._add_byte_tokens()

        for a, b in self.merges:
            new_token = a + b

            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab)

        self.id_to_token = {
            token_id: token
            for token, token_id in self.vocab.items()
        }

        print(f"Tokenizer loaded: {path}")
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Merge rules: {len(self.merges)}")

        if len(self.vocab) != self.vocab_size:
            raise ValueError(
                f"Vocabulary mismatch: expected {self.vocab_size}, "
                f"reconstructed {len(self.vocab)}"
            )


# --------------------------------------------------
# 5. LOAD TOKENIZER
# --------------------------------------------------

if not TOKENIZER_FILE.exists():
    raise FileNotFoundError(
        f"Tokenizer not found: {TOKENIZER_FILE}"
    )

tokenizer = Tokenizer()

tokenizer.load(TOKENIZER_FILE)


# Verify tokenizer round-trip.
test_prompt = "The scientist discovered"

test_ids = tokenizer.encode(test_prompt)
decoded_prompt = tokenizer.decode(test_ids)

print("\nTokenizer verification:")
print(f"Original: {test_prompt}")
print(f"Decoded:  {decoded_prompt}")
print(f"Match:    {test_prompt == decoded_prompt}")

if test_prompt != decoded_prompt:
    raise ValueError(
        "Tokenizer round-trip failed. Check tokenizer.json."
    )


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

print("\nBest model checkpoint loaded successfully.")


# --------------------------------------------------
# 7. SAMPLE NEXT TOKEN
# --------------------------------------------------

def sample_next_token(
    logits,
    temperature=0.8,
    top_k=40
):
    if temperature <= 0:
        raise ValueError(
            "Temperature must be greater than zero."
        )

    logits = logits.clone() / temperature

    # Never generate padding or unknown tokens.
    logits[PAD_TOKEN_ID] = float("-inf")
    logits[UNK_TOKEN_ID] = float("-inf")

    # Greedy decoding.
    if top_k <= 1:
        return torch.argmax(logits).item()

    # Keep only the top-k candidates.
    k = min(top_k, logits.numel())

    top_values, top_indices = torch.topk(logits, k)

    filtered_logits = torch.full_like(
        logits,
        float("-inf")
    )

    filtered_logits[top_indices] = top_values

    probabilities = torch.softmax(
        filtered_logits,
        dim=-1
    )

    next_token = torch.multinomial(
        probabilities,
        num_samples=1
    )

    return next_token.item()


# --------------------------------------------------
# 8. GENERATE TEXT
# --------------------------------------------------

@torch.inference_mode()
def generate_text(
    prompt,
    max_new_tokens=200,
    temperature=0.8,
    top_k=40
):
    prompt_ids = tokenizer.encode(prompt)

    if not prompt_ids:
        raise ValueError(
            "Prompt must contain at least one token."
        )

    generated_ids = prompt_ids.copy()

    for _ in range(max_new_tokens):
        # Keep the most recent context.
        context_ids = generated_ids[-MAX_SEQ_LEN:]

        input_tensor = torch.tensor(
            context_ids,
            dtype=torch.long,
            device=DEVICE
        )

        logits = model(input_tensor)

        # Final position predicts the next token.
        next_token_logits = logits[-1]

        next_token_id = sample_next_token(
            next_token_logits,
            temperature=temperature,
            top_k=top_k
        )

        # Stop when EOS is predicted.
        if next_token_id == EOS_TOKEN_ID:
            break

        generated_ids.append(next_token_id)

    # Decode the entire sequence to preserve the prompt.
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
            result = generate_text(
                prompt=prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_k=TOP_K
            )

            print("\nGenerated text:")
            print("-" * 50)
            print(result)
            print("-" * 50)

        except Exception as error:
            print(f"Generation failed: {error}")