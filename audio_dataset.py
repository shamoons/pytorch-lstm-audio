import glob
import random
import torch
import math
import numpy as np
import os.path as path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils.audio_util import load_audio_spectrogram


class AudioDataset(Dataset):
    def __init__(self, corrupted_path, mask, feature_dim=5, train_set=False, test_set=False, normalize=False, repeat_sample=1):
        torch.manual_seed(0)

        self.feature_dim = feature_dim
        self.normalize = normalize
        self.mask = mask

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

        if not self.mask:
            clean_audio_file_paths = np.array(
                sorted(glob.iglob(clean_base_path + '/**/*.flac', recursive=True)))

            x_train, x_test, y_train, y_test = train_test_split(
                corrupted_audio_file_paths, clean_audio_file_paths, test_size=0.1)
        else:
            np.random.shuffle(corrupted_audio_file_paths)
            cutoff = len(corrupted_audio_file_paths)
            x_train = corrupted_audio_file_paths[0:int(cutoff * 0.9)]
            x_test = corrupted_audio_file_paths[int(cutoff * 0.9):]

        if train_set:
            if not self.mask:
                self.clean_file_paths = y_train
            self.corrupted_file_paths = x_train

        if test_set:
            if not self.mask:
                self.clean_file_paths = y_test
            self.corrupted_file_paths = x_test


        if not self.mask:
            self.clean_file_paths = np.repeat(
                self.clean_file_paths, repeat_sample)
        self.corrupted_file_paths = np.repeat(
            self.corrupted_file_paths, repeat_sample)

    def __len__(self):
        return min(len(self.corrupted_file_paths), 1000000)

    def __getitem__(self, index):
        corrupted_file_path = self.corrupted_file_paths[index]

        # corrupted_file_path = '/home/shamoon/speech-enhancement-asr/data/LibriSpeech/dev-noise-subtractive-250ms-1/84/121123/84-121123-0000.flac'

        input_spectrogram, _, _, _, _ = load_audio_spectrogram(
            corrupted_file_path, normalize_spect=self.normalize)

        if not self.mask:
            clean_file_path = self.clean_file_paths[index]
            # clean_file_path = '/home/shamoon/speech-enhancement-asr/data/LibriSpeech/dev-clean/84/121123/84-121123-0000.flac'
            output_spectrogram, _, _, _, _ = load_audio_spectrogram(
                clean_file_path, normalize_spect=self.normalize)
        else:
            mask_filepath = path.splitext(corrupted_file_path)[
                0] + '-mask.npy'
            mask = np.loadtxt(mask_filepath, dtype=np.float32)

            width = input_spectrogram.size(0)
            max_size = (len(mask) // width) * width
            mask_vector = mask[:max_size].reshape(width, -1).max(axis=1)

        input_sliced = input_spectrogram
        if self.mask:
            output_sliced = mask_vector
        else:
            output_sliced = output_spectrogram

        output_sliced = torch.Tensor(output_sliced)

        return input_sliced, output_sliced


def pad_samples(batched_data):
    (xx, yy) = zip(*batched_data)

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)

    return xx_pad, yy_pad
