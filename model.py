import math

from keras import optimizers
from keras.layers import Input, LSTM, Dense, TimeDistributed, Activation, BatchNormalization, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from wandb.keras import WandbCallback


class SpeechBaselineModel():
    def __init__(self, total_samples):
        self.model = Sequential()
        self.total_samples = total_samples
        return

    def build(self, seq_length, feature_dim, lstm1_size, lstm2_size, lstm3_size, lstm4_size):

        self.model.add(LSTM(lstm1_size, input_shape=(
            seq_length, feature_dim), return_sequences=True))
        # self.model.add(Dropout(0.2))
        self.model.add(LSTM(lstm2_size, return_sequences=True))
        # self.model.add(Dropout(0.2))
        self.model.add(LSTM(lstm3_size, return_sequences=True))
        # self.model.add(Dropout(0.2))
        self.model.add(LSTM(lstm4_size, return_sequences=True))
        # self.model.add(Dropout(0.2))
        self.model.add(TimeDistributed(Dense(feature_dim, activation='relu')))

    def compile(self, learning_rate):
        adam_optimizer = optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(loss='mean_squared_error', optimizer=adam_optimizer)

    def train(self, train_gen, val_gen, batch_size, epochs, worker_count, max_queue_size, use_multiprocessing):

        callbacks = [WandbCallback(), EarlyStopping(
            monitor='val_loss', patience=10)]

        self.model.fit_generator(
            train_gen, steps_per_epoch=math.ceil(self.total_samples / batch_size), callbacks=callbacks, epochs=epochs, workers=worker_count, max_queue_size=max_queue_size, use_multiprocessing=use_multiprocessing, validation_data=val_gen)
