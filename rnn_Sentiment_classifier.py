

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout


# 1. Load and Prepare the Data

VOCAB_SIZE = 10000     # only consider top 10,000 most frequent words
MAX_LEN = 200           # pad/truncate every review to 200 words

print("Loading IMDB dataset...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)
print(f"Training samples: {len(x_train)}, Testing samples: {len(x_test)}")

# Pad sequences so every review has the same length (required for RNNs)
x_train = pad_sequences(x_train, maxlen=MAX_LEN)
x_test = pad_sequences(x_test, maxlen=MAX_LEN)


# 2. Build the RNN (LSTM) Model

model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=32, input_length=MAX_LEN),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')   # binary output: positive (1) or negative (0)
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()


# 3. Train the Model

print("\nTraining model...")
history = model.fit(
    x_train, y_train,
    epochs=5,                 # keep small so it trains fast on CPU
    batch_size=128,
    validation_split=0.2,
    verbose=1
)


# 4. Evaluate on Test Data

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")


# 5. Plot Training History 

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
print("\nSaved plot as training_history.png -- use this in your report!")


# 6. Tried it on a custom review 

word_index = imdb.get_word_index()

def predict_sentiment(review_text):
    words = review_text.lower().split()
    encoded = [word_index.get(w, 2) + 3 for w in words]  # +3 offset used by Keras IMDB
    padded = pad_sequences([encoded], maxlen=MAX_LEN)
    prediction = model.predict(padded, verbose=0)[0][0]
    sentiment = "Positive" if prediction >= 0.5 else "Negative"
    print(f"Review: \"{review_text}\"\nPredicted Sentiment: {sentiment} (score: {prediction:.3f})")

print("\n--- Demo Prediction ---")
predict_sentiment("this movie was absolutely fantastic and heartwarming")
predict_sentiment("what a terrible waste of time, i hated every minute")


model.save('sentiment_model.h5')
print("\nModel saved as sentiment_model.h5 -- ready for deployment!")
 
import json
with open('word_index.json', 'w') as f:
    json.dump(word_index, f)
print("Saved word_index.json -- needed by the Streamlit app.")