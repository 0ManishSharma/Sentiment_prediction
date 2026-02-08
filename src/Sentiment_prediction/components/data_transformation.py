import os
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class DataTransformation:
    def __init__(self):
        self.max_words = 10000
        self.max_len = 200
        self.tokenizer_path = "artifacts/tokenizer.pkl"

    def clean_text(self, text):
        text = str(text).lower()
        return text

    def initiate_transformation(self):
        train_df = pd.read_csv("artifacts/train.csv")
        test_df = pd.read_csv("artifacts/test.csv")

        # ✅ Clean text
        train_df["review"] = train_df["review"].apply(self.clean_text)
        test_df["review"] = test_df["review"].apply(self.clean_text)

        # ✅ SAME dataframe → X & y
        X_train = train_df["review"].values
        y_train = train_df["sentiment"].map({"positive": 1, "negative": 0}).values

        X_test = test_df["review"].values
        y_test = test_df["sentiment"].map({"positive": 1, "negative": 0}).values

        # ✅ Tokenizer
        tokenizer = Tokenizer(num_words=self.max_words, oov_token="<OOV>")
        tokenizer.fit_on_texts(X_train)

        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_test_seq = tokenizer.texts_to_sequences(X_test)

        X_train_pad = pad_sequences(X_train_seq, maxlen=self.max_len, padding="post")
        X_test_pad = pad_sequences(X_test_seq, maxlen=self.max_len, padding="post")

        with open(self.tokenizer_path, "wb") as f:
            pickle.dump(tokenizer, f)

        print("✅ Data transformation completed")
        print("Train size:", X_train_pad.shape, y_train.shape)
        print("Test size:", X_test_pad.shape, y_test.shape)

        return X_train_pad, y_train, X_test_pad, y_test
