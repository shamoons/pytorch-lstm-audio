import random
import glob
import math
import numpy as np
import os.path as path
from audio_util import load_audio_spectrogram
from keras.utils import Sequence


class DataGenerator(Sequence):
    def __init__(self, corrupted_path, seq_length=10, batch_size=20, train_set=False, test_set=False):
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

    def count_files(self):
        return len(self.clean_file_paths)

    def __getitem__(self, index):
        batch_index = index * self.batch_size

        input_spectrogram = load_audio_spectrogram(
            self.corrupted_file_paths[batch_index])

        output_spectrogram = load_audio_spectrogram(
            self.clean_file_paths[batch_index])

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
