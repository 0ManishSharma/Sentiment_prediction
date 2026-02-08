from dataclasses import dataclass
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense,Dropout,LSTM,Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

@dataclass
class ModelConfig:
    model_file_path = "artifacts/model.h5"
class SentimentModel:
    def __init__(self):
        self.config = ModelConfig()
        os.makedirs(os.path.dirname(self.config.model_file_path),exist_ok=True)

    def build_model(self):
        model = Sequential([
            Embedding(
                input_dim=10000,
                output_dim = 128,
                input_length = 200
            ),
            LSTM(128),
            Dropout(0.5),
            Dense(1,activation="sigmoid")
        ])
        model.compile(
            optimizer = Adam(learning_rate=0.001),
            loss ="binary_crossentropy",
            metrics = ['accuracy']
        )
        return model
    def train(self,X_train,X_test,y_train,y_test):
        model = self.build_model()

        early_stoping = EarlyStopping(
            monitor = "val_loss",
            patience = 2,
            restore_best_weights = True
        )
        model_checkpoint= ModelCheckpoint(
            filepath = self.config.model_file_path,
            monitor = "val_loss",
            save_best_only = True
        )

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=10,
            batch_size=64,
            callbacks=[early_stoping, model_checkpoint],
            verbose=1
        )

        print("✅ Model training completed")
        print("📦 Best model saved at:", self.config.model_file_path)

        return model, history

        