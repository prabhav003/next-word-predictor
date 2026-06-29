import streamlit as st
import torch
import torch.nn as nn
import pickle
import re
import string
from nltk.tokenize import word_tokenize
import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab")

# load vocab
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

with open("idx_to_word.pkl", "rb") as f:
    idx_to_word = pickle.load(f)

with open("len_list.pkl", "rb") as f:
    len_list = pickle.load(f)
max_len = max(len_list)

# creating_model
class LSTMModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 100)
        self.lstm = nn.LSTM(100, 150, batch_first=True)
        self.fc = nn.Linear(150, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, cell) = self.lstm(embedded)
        output = self.fc(hidden.squeeze(0))
        return output

# load model   
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_SIZE = max(vocab.values()) + 1
model = LSTMModel(len(vocab))
model.load_state_dict(
    torch.load("lstm_next_word.pth", map_location=device)
)
model.to(device)
model.eval()

# functions
def preprocess(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )
    return text

def text_to_indices(tokens):
    indices = []
    for token in tokens:
        if token in vocab:
            indices.append(vocab[token])
    return indices

def predict(text):
    text = preprocess(text)
    tokens = word_tokenize(text)
    indices = text_to_indices(tokens)
    if len(indices) == 0:
        return "<words_not_found>"

    # keep only last max_len-1 words
    indices = indices[-(max_len - 1):]
    padding = [0] * ((max_len - 1) - len(indices))
    padded = padding + indices
    padded = torch.tensor(
        padded,
        dtype=torch.long
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(padded)
        predicted_index = torch.argmax(output, dim=1).item()
    predicted_word = idx_to_word[predicted_index]
    return predicted_word

# ui

st.title("🧠 Next Word Predictor using LSTM")

st.write("Enter a sentence and the model will predict the next word.")

text = st.text_input(
    "Input Sentence"
)

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter a sentence.")
    else:
        word = predict(text)
        st.success(f"Predicted Next Word: **{word}**")

