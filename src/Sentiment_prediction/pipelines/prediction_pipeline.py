import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

class PredictionPipeline:
    def __init__(self):
        self.model = load_model("artifacts/model.h5")
        with open("artifacts/tokenizer.pkl", "rb") as f:
            self.tokenizer = pickle.load(f)

        self.max_len = 200

    def predict(self, text):
        seq = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=self.max_len, padding="post")
        result = self.model.predict(padded)[0][0]

        return "Positive 😊" if result > 0.5 else "Negative 😠"
