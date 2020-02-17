import glob
import wandb
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import math
import random
import os
import socket
import multiprocessing
import os.path as path
import soundfile as sf
import argparse

from scipy import signal

from sklearn.preprocessing import normalize
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, LSTM, Dense, TimeDistributed, Activation, BatchNormalization, Dropout
from keras.models import Sequential
from keras.utils import Sequence
from keras import optimizers
from wandb.keras import WandbCallback

TOTAL_SAMPLES = 2676
SEQ_LENGTH = 100
VECTOR_SIZE = 129


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
            start_index = random.randint(
                0, len(input_spectrogram) - self.seq_length)
            end_index = start_index + self.seq_length

            input_sliced = input_spectrogram[start_index:end_index]
            output_sliced = output_spectrogram[start_index:end_index]

            inputs.append(input_sliced)
            outputs.append(output_sliced)

        inputs_array = np.array(inputs)
        outputs_array = np.array(outputs)

        normalized_inputs = (
            inputs_array - inputs_array.mean()) / inputs_array.std()

        normalized_outputs = (
            outputs_array - outputs_array.mean()) / outputs_array.std()

        return normalized_inputs, normalized_outputs


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--LSTM_1_SIZE", help="Hidden size for the first LSTM Layer", type=int, default=256)

    parser.add_argument(
        "--LSTM_2_SIZE", help="Hidden size for the second LSTM Layer", type=int, default=256)

    parser.add_argument(
        "--LSTM_3_SIZE", help="Hidden size for the third LSTM Layer", type=int, default=256)

    parser.add_argument(
        "--LSTM_4_SIZE", help="Hidden size for the fourth LSTM Layer", type=int, default=256)

    parser.add_argument('--learning_rate', help='Learning rate for optimizer',
                        type=float, default=0.01)

    parser.add_argument('--epochs', help='Epochs to run',
                        type=int, default=250)

    parser.add_argument('--batch_size', help='Batch size',
                        type=int, default=32)

    parser.add_argument('--worker_count', help='Number of workers for fit_generator',
                        type=int, default=multiprocessing.cpu_count())

    parser.add_argument('--max_queue_size', help='Max queue size for fit_generator',
                        type=int, default=32 * 8)

    parser.add_argument('--use_multiprocessing', help='Use multiprocessing for fit_generator',
                        type=bool, default=False)

    args = parser.parse_args()

    return args


def main():
    wandb_tags = [socket.gethostname()]
    wandb.init(project="pytorch-lstm-audio", tags=','.join(wandb_tags))

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    tf.random.set_seed(0)
    np.random.seed(0)

    args = parse_args()

    model = Sequential()
    model.add(LSTM(args.LSTM_1_SIZE, input_shape=(
        SEQ_LENGTH, VECTOR_SIZE), return_sequences=True))
    # model.add(Dropout(0.2))
    model.add(LSTM(args.LSTM_2_SIZE, return_sequences=True))
    # model.add(Dropout(0.2))
    model.add(LSTM(args.LSTM_3_SIZE, return_sequences=True))
    # model.add(Dropout(0.2))
    model.add(LSTM(args.LSTM_4_SIZE, return_sequences=True))
    # model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(VECTOR_SIZE, activation='relu')))

    adam_optimizer = optimizers.Adam(learning_rate=args.learning_rate)
    model.compile(loss='mean_squared_error',
                  optimizer=adam_optimizer)

    print(model.summary())

    callbacks = [WandbCallback(), EarlyStopping(
        monitor='val_loss', patience=10), ModelCheckpoint(filepath='saved_models/' + 'model' + '.hdf5', monitor='val_loss', save_best_only=True)]

    trainGen = DataGenerator(
        'data/dev-noise-subtractive-250ms-1', seq_length=SEQ_LENGTH, batch_size=args.batch_size, train_set=True)
    valGen = DataGenerator(
        'data/dev-noise-subtractive-250ms-1', seq_length=SEQ_LENGTH, batch_size=args.batch_size, test_set=True)

    model.fit_generator(
        trainGen, steps_per_epoch=math.ceil(TOTAL_SAMPLES / args.batch_size), callbacks=callbacks, epochs=args.epochs, workers=args.worker_count, max_queue_size=args.max_queue_size, use_multiprocessing=args.use_multiprocessing, validation_data=valGen)


if __name__ == '__main__':
    main()
