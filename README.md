# Movie Review Sentiment Classifier 
An LSTM-based sentiment classifier trained on the IMDB movie reviews dataset, deployed as an interactive Streamlit web app. Built as part of a 45-Day AI Training Course project.

**Features**
Binary sentiment classification (Positive/Negative) using an Embedding + LSTM architecture
~85% test accuracy on the IMDB dataset
Real-time predictions via an interactive Streamlit interface
Evaluated on custom real-world sentences, including negation, sarcasm, and mixed-sentiment cases
**Tech Stack**
Python
TensorFlow / Keras
Streamlit
**How to Run**
Install dependencies:
   pip install tensorflow numpy matplotlib streamlit
Train the model (this also saves sentiment_model.h5 and word_index.json):
   python rnn_sentiment_classifier.py
Launch the web app:
   streamlit run app.py
Open the local URL shown in the terminal (usually http://localhost:8501) in your browser.
**Model Architecture**

Embedding (32-dim) → LSTM (64 units) → Dropout (0.5) → Dense (32, ReLU) → Dense (1, Sigmoid)

**Notes**

The trained model file (.h5) is not included in this repo due to size — run the training script to regenerate it locally.
