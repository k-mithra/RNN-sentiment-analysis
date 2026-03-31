import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

st.title("IMDB Sentiment Analysis (RNN)")

# Load TF-IDF
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# Preprocessing functions
def clean_text(text):
    text = text.lower()
    text = re.sub(r"https\S+","",text)
    text = re.sub(r"<.*?>","",text)
    text = re.sub(r"[^A-Za-z0-9\s]","",text)

    tokens = word_tokenize(text)
    stop_words = stopwords.words("english")
    tokens = [word for word in tokens if word not in stop_words]

    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens]

    return " ".join(tokens)

# Model class
class RNN(nn.Module):
    def __init__(self,input_size,hidden_size=128,num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_size,1)

    def forward(self,x):
        h0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size)
        out,_ = self.rnn(x,h0)
        out = self.fc(out[:,-1,:])
        return out

# Load model
input_size = 5000   # TF-IDF max_features
model = RNN(input_size)
model.load_state_dict(torch.load("rnn_model.pt", map_location=torch.device('cpu')))
model.eval()

# User input
text = st.text_area("Enter Movie Review")

if st.button("Predict"):
    cleaned = clean_text(text)

    vector = tfidf.transform([cleaned]).toarray()

    tensor = torch.tensor(vector, dtype=torch.float32).unsqueeze(1)

    output = model(tensor)
    prob = torch.sigmoid(output).item()

    if prob > 0.5:
        st.success("Positive Review 😊")
    else:
        st.error("Negative Review 😞")