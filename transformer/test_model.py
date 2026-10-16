import torch

from next_word_predictor import NextWordPredictor


def main():

    # Must match the tokenizer's actual vocabulary size.
    vocab_size = 1000

    model = NextWordPredictor(
        vocab_size=vocab_size,
        embedding_dim=256,
        num_heads=8,
        ff_dim=1024,
        num_layers=4,
        max_seq_len=128
    )

    # Test 1: Single sequence.

    token_ids = torch.tensor(
        [10, 20, 30, 40, 50],
        dtype=torch.long
    )

    logits = model(token_ids)

    assert logits.shape == (5, vocab_size)

    print("Single sequence:", logits.shape)

    # Test 2: Batch of sequences.

    batch = torch.tensor(
        [
            [10, 20, 30, 40, 50],
            [11, 21, 31, 41, 51]
        ],
        dtype=torch.long
    )

    logits = model(batch)

    assert logits.shape == (2, 5, vocab_size)

    print("Batch:", logits.shape)

    # Test 3: Next-token prediction.

    next_token, probabilities = model.predict_next_token(
        token_ids
    )

    assert 0 <= next_token < vocab_size

    assert probabilities.shape == (vocab_size,)

    print("Predicted token ID:", next_token)
    print("Probability sum:", probabilities.sum().item())

    # Test 4: Gradient propagation.

    model.train()

    logits = model(batch)

    loss = logits.mean()

    loss.backward()

    print("Gradient propagation: successful")

    print("\nAll basic model tests passed.")


if __name__ == "__main__":
    main()