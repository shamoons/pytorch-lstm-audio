import random
import glob
import math
import tensorflow as tf
import numpy as np
import os.path as path
from audio_util import load_audio_spectrogram, load_mel_spectrogram
from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):
    def __init__(self, corrupted_path, seq_length=10, batch_size=20, train_set=False, test_set=False, repeat_sample=10, n_mels=None):
        corrupted_base_path = path.abspath(corrupted_path)
        corrupted_base_path_parts = corrupted_base_path.split('/')
        clean_base_path = corrupted_base_path_parts.copy()
        if 'dev' in clean_base_path[-1]:
            clean_base_path[-1] = 'dev-clean'
        elif 'train' in clean_base_path[-1]:
            clean_base_path[-1] = 'train-clean'
        elif 'test' in clean_base_path[-1]:
            clean_base_path[-1] = 'test-clean'

        clean_base_path = '/'.join(clean_base_path)

        corrupted_audio_file_paths = np.array(
            sorted(glob.iglob(corrupted_base_path + '/**/*.flac', recursive=True)))

        clean_audio_file_paths = np.array(
            sorted(glob.iglob(clean_base_path + '/**/*.flac', recursive=True)))

        cutoff_index = int(len(corrupted_audio_file_paths) * 0.9)

        if train_set == True:
            self.clean_file_paths = clean_audio_file_paths[0: cutoff_index]
            self.corrupted_file_paths = corrupted_audio_file_paths[0: cutoff_index]
        if test_set == True:
            self.clean_file_paths = clean_audio_file_paths[cutoff_index:]
            self.corrupted_file_paths = corrupted_audio_file_paths[cutoff_index:]

        self.clean_file_paths = np.repeat(self.clean_file_paths, repeat_sample)
        self.corrupted_file_paths = np.repeat(
            self.corrupted_file_paths, repeat_sample)
        self.clean_file_paths = self.clean_file_paths[0:100]
        self.corrupted_file_paths = self.corrupted_file_paths[0:100]

        self.seq_length = seq_length
        self.batch_size = batch_size
        self.n_mels = n_mels
        return

    def __len__(self):
        return math.ceil(len(self.clean_file_paths) / self.batch_size)

    def count_files(self):
        return len(self.clean_file_paths)

    def __getitem__(self, index):
        inputs_array = tf.random.uniform(
            shape=(64, 100, 128),
            minval=-1,
            maxval=1
        )

        outputs_array = inputs_array * 2

        return inputs_array, outputs_array, [None]

    def __getitem__2(self, index):
        batch_index = index * self.batch_size

        input_spectrogram = load_mel_spectrogram(
            self.corrupted_file_paths[batch_index], n_mels=self.n_mels)

        output_spectrogram = load_mel_spectrogram(
            self.clean_file_paths[batch_index], n_mels=self.n_mels)

        # input_spectrogram = load_audio_spectrogram(
        #     self.corrupted_file_paths[batch_index])

        # output_spectrogram = load_audio_spectrogram(
        #     self.clean_file_paths[batch_index])

        inputs = []
        outputs = []

        while len(inputs) < self.batch_size:
            if len(input_spectrogram) - self.seq_length < 0:
                continue
            start_index = random.randint(
                0, len(input_spectrogram) - self.seq_length)
            end_index = start_index + self.seq_length

            input_sliced = input_spectrogram[start_index:end_index]
            output_sliced = output_spectrogram[start_index:end_index]

            inputs.append(input_sliced)
            outputs.append(output_sliced)

        inputs_array = np.array(inputs)
        outputs_array = np.array(outputs)

        inputs_array = (inputs_array - inputs_array.mean()) / \
            inputs_array.std()
        outputs_array = (outputs_array - outputs_array.mean()
                         ) / outputs_array.std()

        # Added None because of https://stackoverflow.com/a/60131716/239879
        return inputs_array, outputs_array, [None]
