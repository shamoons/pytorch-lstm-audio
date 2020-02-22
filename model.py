import math
import os
import tensorflow as tf

from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import multi_gpu_model
from wandb.keras import WandbCallback
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
import wandb


class SpeechBaselineModel():
    def __init__(self, total_samples):
        self.model = Sequential()
        self.total_samples = total_samples
        return

    def build(self, seq_length, feature_dim, lstm1_size, lstm2_size, lstm3_size, lstm4_size):
        if tf.test.is_gpu_available(cuda_only=True) == True:
            LSTMLayer = CuDNNLSTM
        else:
            LSTMLayer = LSTM

        self.model.add(LSTMLayer(lstm1_size, input_shape=(
            seq_length, feature_dim), return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTMLayer(lstm2_size, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTMLayer(lstm3_size, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTMLayer(lstm4_size, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(feature_dim, activation='linear'))

    def compile(self, learning_rate):
        try:
            self.model = multi_gpu_model(self.model)
        except:
            pass

        adam_optimizer = optimizers.Adam(
            learning_rate=learning_rate, clipnorm=1.0)
        self.model.compile(loss='mean_squared_error', optimizer=adam_optimizer)

    def train(self, train_gen, val_gen, batch_size, epochs, worker_count, max_queue_size, use_multiprocessing):

        callbacks = [WandbCallback(save_weights_only=False, monitor='val_loss'), EarlyStopping(
            monitor='val_loss', patience=10)]

        return self.model.fit(
            train_gen, steps_per_epoch=math.ceil(self.total_samples / batch_size), callbacks=callbacks, epochs=epochs, workers=worker_count, max_queue_size=max_queue_size, use_multiprocessing=use_multiprocessing, validation_data=val_gen)

    def predict(self, X):
        pred = self.model.predict(X)
        return pred
