import streamlit as st
import torch
import torch.nn as nn
import pickle
from nltk.tokenize import word_tokenize
import nltk
nltk.download("punkt", quiet=True)

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
        self.lstm = nn.LSTM(
            input_size=100,
            hidden_size=150,
            batch_first=True
        )
        self.fc = nn.Linear(150, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        x = self.fc(hidden.squeeze(0))
        return x
    

# load model   
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_SIZE = max(vocab.values()) + 1
model = LSTMModel(VOCAB_SIZE)
model.load_state_dict(
    torch.load(
        "lstm_next_word.pth",
        map_location=device
    )
)
model.to(device)
model.eval()


# function
def text_to_indices(tokens, vocab):
    numerical_sentence = []
    for token in tokens:
        if token in vocab:
            numerical_sentence.append(vocab[token])
        else:
            numerical_sentence.append(vocab["<unk>"])
    return numerical_sentence

# prediction

def predict(text):

    # SAME preprocessing as training

    text = text.lower()
    tokens = word_tokenize(text)
    numerical_text = text_to_indices(tokens, vocab)
    # Keep only last max_len-1 tokens
    numerical_text = numerical_text[-(max_len - 1):]
    padded = [0] * ((max_len - 1) - len(numerical_text))
    padded += numerical_text
    padded = torch.tensor(
        padded,
        dtype=torch.long
    ).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(padded)
        probabilities = torch.softmax(output, dim=1)
        predicted_index = torch.argmax(probabilities, dim=1).item()
    predicted_word = idx_to_word[predicted_index]
    return predicted_word

# ui


st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="🧠"
)
st.title("🧠 Next Word Predictor")
sentence = st.text_input(
    "Enter a sentence"
)
if st.button("Predict"):
    if sentence.strip() == "":
        st.warning("Please enter a sentence.")
    else:
        prediction = predict(sentence)
        st.success(f"Predicted Word : {prediction}")

        