import glob
import wandb
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import math
import random
import os
import multiprocessing
import os.path as path
import soundfile as sf
import argparse

from scipy import signal

from sklearn.preprocessing import normalize
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, LSTM, Dense, TimeDistributed, Activation, BatchNormalization
from keras.models import Sequential
from keras.utils import Sequence
from keras import optimizers
from wandb.keras import WandbCallback

TOTAL_SAMPLES = 2676
SEQ_LENGTH = 100
VECTOR_SIZE = 129
BATCH_SIZE = 32


def get_test_data(batches=1):
    inputs = tf.random.uniform(
        shape=(batches, 10, VECTOR_SIZE),
        minval=-1,
        maxval=1
    )

    outputs = inputs * 2
    return inputs, outputs


def get_training_data(batches=1):
    inputs = tf.random.uniform(
        shape=(batches, 10, VECTOR_SIZE),
        minval=-1,
        maxval=1
    )

    outputs = inputs * 2
    return inputs, outputs


class DataGenerator(Sequence):
    def __init__(self, corrupted_path, seq_length=10, batch_size=20, train_set=False, test_set=False):
        corrupted_base_path = path.abspath(corrupted_path)
        corrupted_base_path_parts = corrupted_base_path.split('/')
        clean_base_path = corrupted_base_path_parts.copy()
        clean_base_path[-1] = 'dev-clean'
        clean_base_path = '/'.join(clean_base_path)

        corrupted_audio_file_paths = list(
            sorted(glob.iglob(corrupted_base_path + '/**/*.flac', recursive=True)))

        clean_audio_file_paths = list(
            sorted(glob.iglob(clean_base_path + '/**/*.flac', recursive=True)))

        cutoff_index = int(len(corrupted_audio_file_paths) * 0.9)

        if train_set == True:
            self.clean_file_paths = clean_audio_file_paths[0: cutoff_index]
            self.corrupted_file_paths = corrupted_audio_file_paths[0: cutoff_index]
        if test_set == True:
            self.clean_file_paths = clean_audio_file_paths[cutoff_index:]
            self.corrupted_file_paths = corrupted_audio_file_paths[cutoff_index:]

        self.seq_length = seq_length
        self.batch_size = batch_size
        return

    def __len__(self):
        return math.ceil(len(self.clean_file_paths) / self.batch_size)

    def __getitem__(self, index):
        batch_index = index * self.batch_size

        corrupted_samples, corrupted_sample_rate = sf.read(
            self.corrupted_file_paths[batch_index])

        clean_samples, clean_sample_rate = sf.read(
            self.clean_file_paths[batch_index])

        _, _, corrupted_spectrogram = signal.spectrogram(
            corrupted_samples, corrupted_sample_rate)

        _, _, clean_spectrogram = signal.spectrogram(
            clean_samples, clean_sample_rate)

        # By default, the first axis is frequencies and the second is time.
        # We swap them here.
        input_spectrogram = np.swapaxes(corrupted_spectrogram, 0, 1)
        output_spectrogram = np.swapaxes(clean_spectrogram, 0, 1)

        inputs = []
        outputs = []

        while len(inputs) < self.batch_size:
            input_sliced = self.__get_random_slice(
                input_spectrogram, self.seq_length)

            output_sliced = self.__get_random_slice(
                output_spectrogram, self.seq_length)

            inputs.append(input_sliced)
            outputs.append(output_sliced)

        inputs_array = np.array(inputs)
        outputs_array = np.array(outputs)

        normalized_inputs = (
            inputs_array - inputs_array.mean()) / inputs_array.std()

        normalized_outputs = (
            outputs_array - outputs_array.mean()) / outputs_array.std()

        return normalized_inputs, normalized_outputs

    def __getitem__2(self, index):
        inputs = []
        outputs = []

        while len(inputs) < self.batch_size:
            inp = tf.random.uniform(
                shape=(self.seq_length, VECTOR_SIZE),
                minval=-1,
                maxval=1
            )

            outp = inp * 2

            inputs.append(inp)
            outputs.append(outp)

        return np.array(inputs), np.array(outputs)

    def __get_random_slice(self, data, slice_length):
        start_index = random.randint(0, len(data) - slice_length)
        end_index = start_index + slice_length

        return data[start_index:end_index]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--LSTM_1_SIZE", help="Hidden size for the first LSTM Layer", type=int, default=256)

    parser.add_argument(
        "--LSTM_2_SIZE", help="Hidden size for the second LSTM Layer", type=int, default=128)

    args = parser.parse_args()

    return args


def main():
    wandb.init(project="pytorch-lstm-audio")

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    tf.random.set_seed(0)
    np.random.seed(0)

    args = parse_args()

    model = Sequential()
    model.add(LSTM(args.LSTM_1_SIZE, input_shape=(
        SEQ_LENGTH, VECTOR_SIZE), return_sequences=True))
    model.add(LSTM(args.LSTM_2_SIZE, return_sequences=True))
    model.add(TimeDistributed(Dense(VECTOR_SIZE, activation='relu')))

    adam_optimizer = optimizers.Adam(
        learning_rate=0.1, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='mean_squared_error',
                  optimizer=adam_optimizer)

    print(model.summary())

    callbacks = [WandbCallback(), EarlyStopping(
        monitor='val_loss', patience=10), ModelCheckpoint(filepath='saved_models/' + 'model' + '.hdf5', monitor='val_loss', save_best_only=True)]

    trainGen = DataGenerator(
        'data/dev-noise-subtractive-250ms-1', seq_length=SEQ_LENGTH, batch_size=BATCH_SIZE, train_set=True)
    valGen = DataGenerator(
        'data/dev-noise-subtractive-250ms-1', seq_length=SEQ_LENGTH, batch_size=BATCH_SIZE, test_set=True)

    worker_count = multiprocessing.cpu_count()
    max_queue_size = BATCH_SIZE * 4
    use_multiprocessing = True
    model.fit_generator(
        trainGen, steps_per_epoch=math.ceil(TOTAL_SAMPLES / BATCH_SIZE), callbacks=callbacks, epochs=250, workers=worker_count, max_queue_size=max_queue_size, use_multiprocessing=use_multiprocessing, validation_data=valGen)


if __name__ == '__main__':
    main()
