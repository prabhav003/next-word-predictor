# Next Word Predictor: Transformer Language Model

This directory contains a small decoder-only Transformer for next-token prediction. The latest experiment is documented in [`nextwordpredictor.ipynb`](nextwordpredictor.ipynb), which was run on Kaggle using a TinyStories text dataset and a Tesla T4 GPU. The Python scripts and checked-in data/checkpoints are an earlier Tiny Shakespeare experiment; their results are reported separately below.

## Kaggle notebook experiment

The notebook reads the `text` column from the TinyStories train and validation CSV files. It learns a byte-level BPE tokenizer from a training-text sample, then encodes the full training and validation splits. The model predicts the next token in story text; it does not perform a story-classification task.

### Model and training setup

- Decoder-only Transformer, **653,800 trainable parameters**
- Embedding dimension 128, 4 attention heads, 2 layers, feed-forward dimension 512
- Context length 128 tokens; BPE vocabulary size 1,000
- Training split: **606,047,153 tokens** across 2,119,489 stories
- Validation split: **6,089,728 tokens** across 21,990 stories
- Batch size 32; AdamW at learning rate 0.0003; gradient clipping at 1.0
- Up to 30 epochs, with at most 500 randomly sampled batches per epoch, a learning-rate scheduler, and early stopping
- Runtime recorded in the notebook: about **378 seconds** on a Tesla T4

### Recorded validation results

| Metric | Result |
|---|---:|
| Best epoch | 30 |
| Training loss at epoch 30 | 2.6013 |
| Validation cross-entropy loss | 2.5976 |
| Validation perplexity | 13.4318 |
| Validation transitions evaluated | 6,089,727 of 6,089,727 |

The notebook does **not** report classification accuracy. For a next-token language model, this run reports cross-entropy loss and perplexity instead. The validation loss fell from 7.0991 before training to 2.5976 after training. These are validation results, not a held-out test result.

The notebook's single displayed generation used the prompt “One morning, a little girl found a mysterious key.” The model immediately generated EOS, so the output contained no continuation. The improved validation perplexity therefore does not, by itself, establish that generated stories are consistently useful or coherent.

### Run the notebook

Open `nextwordpredictor.ipynb` in Kaggle with the required TinyStories CSV dataset attached. The notebook writes token arrays, tokenizer files, checkpoints, training history, and a generated sample under `/kaggle/working/`. Those Kaggle outputs are not included in this repository, so the notebook must be rerun to recreate them.

## Local Tiny Shakespeare scripts

The standalone scripts use the checked-in `data/` files and train/evaluate the earlier Shakespeare model. Run them from this directory:

```bash
python3 preprocessing.py
python3 train.py
python3 evaluate.py
python3 generate.py
```

The checked-in evaluation reports test loss **3.9165**, test perplexity **50.22**, and **48,640** evaluated test targets. The final 19 transitions are dropped because the evaluator uses fixed 128-token chunks. These local results are separate from the Kaggle TinyStories notebook results above.

## Architecture

The model is assembled from modules in this directory:

- `embeddings.py`: learned token embeddings
- `positional_encoding.py`: sinusoidal position encodings
- `masked_multihead_attention.py`: causal multi-head self-attention
- `add_norm.py` and `ffn.py`: residual normalization and feed-forward layers
- `next_word_predictor.py`: decoder blocks and next-token output logits

## Limitations and next steps

- The Kaggle run uses a large corpus, but samples at most 500 batches per epoch; it does not exhaust the full training split each epoch.
- There is no held-out TinyStories test evaluation or baseline comparison in the notebook.
- Generation has only one recorded prompt, and that sample terminated immediately. Evaluate multiple prompts and tune EOS/sampling behavior before drawing conclusions about generation quality.
- The notebook contains its own tokenizer and model definitions alongside the standalone modules. Keeping these implementations in sync would reduce maintenance errors.
- The local scripts use the older Shakespeare data and should not be treated as evidence for the TinyStories notebook's performance.
