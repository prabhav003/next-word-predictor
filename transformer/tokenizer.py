'''

BYTE PAIR ENCODING

'aaabdaaabac': most_frequent_pair = 'aa'

=>
'ZabdZabac' : Z=aa

=>
'ZYdZYac' : Z=aa , Y=ab

=>
'XdXac' : Z=aa , Y=ab , X=ZY

'''

import json
from collections import Counter
from pathlib import Path

import numpy as np


class Tokenizer:
    def __init__(self, vocab_size=1000):
        if vocab_size < 259:
            raise ValueError("vocab_size must be at least 259.")

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
        """Add all 256 possible byte values to the vocabulary."""
        for i in range(256):
            token = bytes([i])

            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

        self.id_to_token = {
            token_id: token
            for token, token_id in self.vocab.items()
        }

    def get_pairs(self, tokens):
        """Count adjacent token pairs."""
        return Counter(zip(tokens, tokens[1:]))

    def merge_pair(self, tokens, pair):
        """Merge every occurrence of the selected pair."""
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

    def fit(self, text):
        """Learn BPE merges using training text only."""
        tokens = [
            bytes([b])
            for b in text.encode("utf-8")
        ]

        self.merges = []
        self.vocab = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<EOS>": 2
        }

        self._add_byte_tokens()

        while len(self.vocab) < self.vocab_size:
            pairs = self.get_pairs(tokens)

            if not pairs:
                break

            pair, frequency = pairs.most_common(1)[0]

            # Ignore pairs that occur only once.
            if frequency < 2:
                break

            new_token = pair[0] + pair[1]

            # Merge the selected pair throughout the corpus.
            tokens = self.merge_pair(tokens, pair)
            self.merges.append(pair)

            # Avoid assigning another ID to an existing token.
            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab)

        self.id_to_token = {
            token_id: token
            for token, token_id in self.vocab.items()
        }

        print("Tokenizer training complete.")
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Number of merges: {len(self.merges)}")

    def tokenize(self, text):
        """Convert text into byte-level BPE tokens."""
        tokens = [
            bytes([b])
            for b in text.encode("utf-8")
        ]

        for pair in self.merges:
            tokens = self.merge_pair(tokens, pair)

        return tokens

    def encode(self, text):
        """Convert text into integer token IDs."""
        tokens = self.tokenize(text)

        return [
            self.vocab.get(token, self.vocab["<UNK>"])
            for token in tokens
        ]

    def encode_document(self, text):
        """Encode text and append EOS at the document boundary."""
        return self.encode(text) + [self.vocab["<EOS>"]]

    def decode(self, ids):
        """Convert token IDs back into readable text."""
        byte_tokens = []

        for token_id in ids:
            token = self.id_to_token.get(token_id)

            if token is None:
                raise ValueError(f"Unknown token ID: {token_id}")

            if token_id == self.vocab["<PAD>"]:
                continue

            if token_id == self.vocab["<EOS>"]:
                byte_tokens.append(b"\n\n")

            elif token_id == self.vocab["<UNK>"]:
                byte_tokens.append("�".encode("utf-8"))

            elif isinstance(token, bytes):
                byte_tokens.append(token)

        return b"".join(byte_tokens).decode(
            "utf-8",
            errors="replace"
        )

    def save(self, path):
        """Save the vocabulary configuration and BPE merges."""
        data = {
            "vocab_size": self.vocab_size,
            "merges": [
                [a.hex(), b.hex()]
                for a, b in self.merges
            ]
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)

        print(f"Tokenizer saved to {path}")

    def load(self, path):
        """Load BPE merges and reconstruct the vocabulary."""
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

        print(f"Tokenizer loaded from {path}")


def main():
    data_dir = Path(__file__).resolve().parent / "data"

    train_file = data_dir / "train.txt"
    validation_file = data_dir / "validation.txt"
    test_file = data_dir / "test.txt"

    # Check that all datasets exist.
    for file in [train_file, validation_file, test_file]:
        if not file.exists():
            raise FileNotFoundError(f"Dataset not found: {file}")

    # 1. Train the tokenizer using training data only.
    train_text = train_file.read_text(encoding="utf-8")

    tokenizer = Tokenizer(vocab_size=1000)
    tokenizer.fit(train_text)

    # 2. Save the trained tokenizer.
    tokenizer.save(data_dir / "tokenizer.json")

    # 3. Encode each dataset using the same tokenizer.
    for split in ["train", "validation", "test"]:
        file_path = data_dir / f"{split}.txt"
        text = file_path.read_text(encoding="utf-8")

        # Preserve paragraph boundaries using EOS tokens.
        paragraphs = [
            paragraph
            for paragraph in text.split("\n\n")
            if paragraph.strip()
        ]

        token_ids = []

        for paragraph in paragraphs:
            token_ids.extend(
                tokenizer.encode_document(paragraph)
            )

        token_array = np.asarray(token_ids, dtype=np.int32)

        output_file = data_dir / f"{split}_tokens.npy"
        np.save(output_file, token_array)

        print(
            f"{split}: {len(paragraphs)} paragraphs, "
            f"{len(token_array)} tokens"
        )

    # 4. Verify that encoding and decoding work.
    sample = "The cat is sleeping."

    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)

    assert decoded == sample, "Tokenizer round-trip failed!"

    print("\nTokenizer verification successful.")


if __name__ == "__main__":
    main()