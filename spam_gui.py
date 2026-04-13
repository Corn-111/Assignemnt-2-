import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout


data = pd.read_csv("C:/Users/miami/Downloads/archive/spam.csv", encoding="latin-1")[["v1", "v2"]]
data.columns = ["label", "text"]
data["Label"] = data["label"].map({"ham":0, "spam":1})

texts = data["text"].values
labels = data["Label"].values

vocan_size = 1000
max_len = 120

tokenizer = Tokenizer(num_words=vocan_size, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=max_len)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(padded, labels, test_size=0.2, random_state=42)

embedding_dim = 64

model = Sequential([Embedding(vocan_size, embedding_dim),
                    Conv1D(128, 5, activation="relu"),
                    MaxPooling1D(pool_size=4),
                    Bidirectional(LSTM(64)),
                    Dense(64, activation="relu"),
                    Dropout(0.5),
                    Dense(1, activation="sigmoid")])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

model.save("spam_model.h5")

with open("tokenizer.pkl", "wb") as t:
  pickle.dump(tokenizer, t)

print("Training complete")

import tkinter as tk
from tkinter import messagebox
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

model = load_model("spam_model.h5")
with open("tokenizer.pkl", "rb") as t:
  tokenizer = pickle.load(t)

max_len = 120

def predict_spam():
  text = entry.get("1.0",tk.END).strip()
  if not text:
    messagebox.showwarning("Warning", "Enter a message")
    return
  seq = tokenizer.texts_to_sequences([text])
  padded = pad_sequences(seq, maxlen=max_len)
  prediction = model.predict(padded)[0][0]
  if prediction >= 0.5:
    result = "The message is spam"
  else:
    result = "The message is not spam"
  result_label.config(text=result, fg ="Red" if prediction >= 0.5 else "Green")

# GUI window
window = tk.Tk()
window.title("Spam Detection")
window.geometry("600x600")

title = tk.Label(window, text="Spam Detection", font=("Arial", 20, "bold"))
title.pack(pady=10)

entry = tk.Text(window, height=10, width=50)
entry.pack(pady=10)

predict_btn = tk.Button(window, text="Check your message", command=predict_spam, font=("Arial", 15))
predict_btn.pack(pady=10)

result_label = tk.Label(window, text="", font=("Arial", 15))
result_label.pack(pady=10)
window.mainloop()
