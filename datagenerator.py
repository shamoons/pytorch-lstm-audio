import random
import glob
import math
import numpy as np
import os.path as path
from audio_util import load_audio_spectrogram
from keras.utils import Sequence


class DataGenerator(Sequence):
    def __init__(self, corrupted_path, seq_length=10, batch_size=20, train_set=False, test_set=False):
        print('DataGenerator __init__')
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

        # corrupted_samples, sample_rate = sf.read(
        #     self.corrupted_file_paths[batch_index])

        # clean_samples, sample_rate = sf.read(
        #     self.clean_file_paths[batch_index])

        # # Taken from https://github.com/PaddlePaddle/DeepSpeech/blob/766e96e600795cea4187123b9ed76dcd250f2d04/data_utils/featurizer/audio_featurizer.py#L121
        # nperseg = int(sample_rate * 0.001 * 20)  # 20ms

        # _, _, corrupted_spectrogram = signal.spectrogram(
        #     corrupted_samples, fs=sample_rate, nperseg=nperseg, noverlap=nperseg // 2, window=signal.hann(nperseg))

        # _, _, clean_spectrogram = signal.spectrogram(
        #     clean_samples, fs=sample_rate, nperseg=nperseg, noverlap=nperseg // 2, window=signal.hann(nperseg))

        # # By default, the first axis is frequencies and the second is time.
        # # We swap them here.
        # input_spectrogram = np.swapaxes(corrupted_spectrogram, 0, 1)
        # output_spectrogram = np.swapaxes(clean_spectrogram, 0, 1)

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
