import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Conv1D,
    MaxPooling1D,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def prepare_data(vocab_size, max_len, sample_frac=1.0):
    data = pd.read_csv("C:/Users/miami/Downloads/archive/spam.csv", encoding="latin-1")[["v1", "v2"]]
    data.columns = ["label", "text"]
    data["Label"] = data["label"].map({"ham": 0, "spam": 1})

    if sample_frac < 1.0:
        data = data.sample(frac=sample_frac, random_state=SEED)

    texts = data["text"].values
    labels = data["Label"].values

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len)

    X_train, X_test, y_train, y_test = train_test_split(
        padded,
        labels,
        test_size=0.2,
        random_state=SEED,
        stratify=labels,
    )
    return X_train, X_test, y_train, y_test


def build_baseline(vocab_size, embedding_dim=64):
    return Sequential(
        [
            Embedding(vocab_size, embedding_dim),
            Conv1D(128, 5, activation="relu"),
            MaxPooling1D(pool_size=4),
            Bidirectional(LSTM(64)),
            Dense(64, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ]
    )


def build_simpler(vocab_size, embedding_dim=32):
    return Sequential(
        [
            Embedding(vocab_size, embedding_dim),
            GlobalAveragePooling1D(),
            Dense(32, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )


def run_experiment(name, vocab_size, max_len, epochs, sample_frac, builder):
    X_train, X_test, y_train, y_test = prepare_data(vocab_size, max_len, sample_frac)

    model = builder(vocab_size)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        verbose=0,
    )
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    print(
        f"{name}: train_acc={history.history['accuracy'][-1]:.4f}, "
        f"val_acc={history.history['val_accuracy'][-1]:.4f}, test_acc={test_acc:.4f}, "
        f"test_loss={test_loss:.4f}, samples={len(X_train) + len(X_test)}"
    )


if __name__ == "__main__":
    run_experiment(
        name="Exp1_Baseline_FullData",
        vocab_size=1000,
        max_len=120,
        epochs=3,
        sample_frac=1.0,
        builder=build_baseline,
    )

    run_experiment(
        name="Exp2_Baseline_HalfData",
        vocab_size=1000,
        max_len=120,
        epochs=3,
        sample_frac=0.5,
        builder=build_baseline,
    )

    run_experiment(
        name="Exp3_SimplerModel_FullData",
        vocab_size=1000,
        max_len=120,
        epochs=3,
        sample_frac=1.0,
        builder=build_simpler,
    )
