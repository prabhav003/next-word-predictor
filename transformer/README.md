# Tiny Shakespeare Transformer

A small decoder-only Transformer language model built from scratch using PyTorch. Trained on the Tiny Shakespeare dataset to learn next-token prediction and generate Shakespeare-style text.

## Model Architecture

- **Architecture:** Decoder-only Transformer
- **Parameters:** 653,800
- **Embedding dimension:** 128
- **Attention heads:** 4
- **Transformer layers:** 2
- **Feed-forward dimension:** 512
- **Context length:** 128 tokens
- **Vocabulary size:** 1,000

## Training Results

- **Epochs:** 20
- **Training tokens:** 360,072
- **Validation tokens:** 46,872
- **Test tokens:** 48,659
- **Training device:** Apple Silicon MPS

| Metric | Result |
|---|---:|
| Final training loss | 3.5415 |
| Best validation loss | 3.8885 |
| Test loss | 3.9165 |
| Test perplexity | 50.22 |

Training and validation losses decreased throughout the 20 epochs, indicating that the model learned statistical patterns from the training dataset.

## Tech Stack

- Python
- PyTorch
- NumPy

## Usage

Run the following commands from the project directory:

```bash
python preprocessing.py
python train.py
python evaluate.py
python generate.py
```

## Problems and Possible Solutions

- **Test perplexity of 50.22**
  - Experiment with different model configurations, tokenization strategies, and training durations.
  - Evaluate changes using validation loss.

- **Limited training data**
  - Experiment with larger text corpora.
  - Maintain separate training, validation, and test splits.

- **Limited context length (128 tokens)**
  - Experiment with longer context windows.
  - Measure the impact on model performance and memory usage.

- **Potential overfitting**
  - Monitor validation loss during training.
  - Consider early stopping if validation performance stops improving.

- **Generation quality not systematically evaluated**
  - Test multiple prompts.
  - Inspect generated text for coherence, repetition, and premature termination.

- **No baseline comparison**
  - Compare the model against a simple baseline.
  - Use the same dataset and evaluation procedure for fair comparison.

- **19 test tokens not evaluated**
  - Modify the evaluation pipeline to include the final incomplete sequence.
  - Report the exact number of evaluated tokens.

*These are potential issues and improvement directions. Not all have been confirmed as problems, and the proposed solutions have not yet been systematically tested.*

## Future Scope

- Experiment with larger Transformer architectures.
- Compare different tokenization strategies and vocabulary sizes.
- Implement learning-rate scheduling and gradient clipping.
- Experiment with temperature and top-k sampling for text generation.
- Track training experiments and compare model configurations.
- Evaluate generated text using consistent prompts and documented criteria.
- Investigate longer context windows and their impact on next-token prediction.

## Limitations

- The model has a relatively small number of parameters.
- Training is restricted to a small Shakespeare text corpus.
- Generated text quality has not yet been systematically evaluated.
- No baseline comparison has been conducted.
- Test perplexity alone does not establish the coherence or quality of generated text.

## Conclusion

This project implements a small decoder-only Transformer language model using PyTorch and trains it for next-token prediction on the Tiny Shakespeare dataset.

After 20 epochs, the model achieved a training loss of **3.5415**, a best validation loss of **3.8885**, and a test loss of **3.9165**, with a test perplexity of **50.22**.

The project establishes a working training and evaluation pipeline. Future work will focus on improving generation quality, experimenting with model configurations, and conducting systematic comparisons against baseline models.