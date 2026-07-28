import streamlit as st
import json
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
MAX_LEN = 200
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'sentiment_model.h5')
WORD_INDEX_PATH = os.path.join(BASE_DIR, 'word_index.json')
# Load model and word index 
@st.cache_resource
def load_assets():
    model = load_model(MODEL_PATH)
    with open(WORD_INDEX_PATH, 'r') as f:
        word_index = json.load(f)
    return model, word_index
model, word_index = load_assets()
# Prediction function
def predict_sentiment(review_text):
    words = review_text.lower().split()
    encoded = [1]  # start-of-sequence token
    for w in words:
        idx = word_index.get(w)
        if idx is not None and idx < 10000 - 3:
            encoded.append(idx + 3)
        else:
            encoded.append(2)  # out-of-vocabulary token
    padded = pad_sequences([encoded], maxlen=MAX_LEN)
    score = model.predict(padded, verbose=0)[0][0]
    sentiment = "Positive 😊" if score >= 0.5 else "Negative 😞"
    return sentiment, float(score)
# UI Layout
st.set_page_config(page_title="Movie Review Sentiment Classifier", page_icon="🎬")
st.title("🎬 Movie Review Sentiment Classifier")
st.write("An LSTM (RNN) based model")
st.write("Type a movie review below and see if the model thinks it's positive or negative.")
user_input = st.text_area(
    "Enter your movie review:",
    placeholder="e.g. This movie was an absolute masterpiece, I loved every second of it."
) 
if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review first.")
    else:
        sentiment, score = predict_sentiment(user_input)
        st.subheader(f"Prediction: {sentiment}")
        st.progress(score if score <= 1 else 1.0)
        st.write(f"Confidence Score: {score:.3f}") 
st.divider()
st.caption("Model: Embedding + LSTM")
