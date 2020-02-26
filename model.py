import math
import os
import tensorflow as tf

from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import Sequence
from wandb.keras import WandbCallback
import wandb


class SpeechBaselineModel():
    def __init__(self, total_samples):
        self.model = Sequential()
        self.total_samples = total_samples
        return

    def build(self, seq_length, feature_dim, lstm1_size, lstm2_size, lstm3_size, lstm4_size, lstm5_size):

        self.model.add(Bidirectional(LSTM(lstm1_size, input_shape=(
            seq_length, feature_dim), return_sequences=True)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.2))

        self.model.add(Bidirectional(
            LSTM(lstm2_size, return_sequences=True)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.2))

        # BOTTLENECK HERE

        self.model.add(Bidirectional(
            LSTM(lstm3_size, return_sequences=True)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.2))

        self.model.add(Bidirectional(
            LSTM(lstm4_size, return_sequences=True)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.2))

        self.model.add(Bidirectional(
            LSTM(lstm5_size, return_sequences=True)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.2))

        self.model.add(Dense(feature_dim, activation='linear'))

    def compile(self, learning_rate):
        lr_schedule = optimizers.schedules
        adam_optimizer = optimizers.Adam(
            learning_rate=learning_rate, clipnorm=1.0)
        self.model.compile(loss='mean_squared_error', optimizer=adam_optimizer)

    def train(self, train_gen, val_gen, batch_size, epochs, worker_count, max_queue_size, use_multiprocessing):
        early_stopping_patience = min(10, epochs // 10)

        callbacks = [EarlyStopping(
            monitor='val_loss', patience=early_stopping_patience)]
        # callbacks = [WandbCallback(save_weights_only=False, monitor='val_loss'), EarlyStopping(
        #     monitor='val_loss', patience=early_stopping_patience)]

        return self.model.fit(
            train_gen, steps_per_epoch=math.ceil(self.total_samples / batch_size), callbacks=callbacks, epochs=epochs, workers=worker_count, max_queue_size=max_queue_size, use_multiprocessing=use_multiprocessing,            validation_data=val_gen)

    def predict(self, X):
        pred = self.model.predict(X)
        return pred
