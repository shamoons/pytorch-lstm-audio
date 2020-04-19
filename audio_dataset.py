import glob
import torch
import numpy as np
import os.path as path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils.audio_util import load_audio_spectrogram


class AudioDataset(Dataset):
    def __init__(self, corrupted_paths, mask, train_set=False, test_set=False, normalize=False, tune=0):
        torch.manual_seed(0)

        self.normalize = normalize
        self.mask = mask
        self.corrupted_file_paths = []
        self.clean_file_paths = []

        for corrupted_path in corrupted_paths:
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
                    corrupted_audio_file_paths, clean_audio_file_paths, test_size=0.1, random_state=0)
            else:
                np.random.shuffle(corrupted_audio_file_paths)
                cutoff = len(corrupted_audio_file_paths)
                x_train = corrupted_audio_file_paths[0:int(cutoff * 0.9)]
                x_test = corrupted_audio_file_paths[int(cutoff * 0.9):]

            if train_set:
                if not self.mask:
                    self.clean_file_paths.extend(y_train)
                self.corrupted_file_paths.extend(x_train)
            elif test_set:
                if not self.mask:
                    self.clean_file_paths.extend(y_test)
                self.corrupted_file_paths.extend(x_test)

        if tune > 0:
            self.corrupted_file_paths = self.corrupted_file_paths[0:tune]
            self.clean_file_paths = self.clean_file_paths[0:tune]

            print(self.corrupted_file_paths)

    def __len__(self):
        return min(len(self.corrupted_file_paths), 32e10)

    def __getitem__(self, index):
        corrupted_file_path = self.corrupted_file_paths[index]

        # corrupted_file_path = '/home/shamoon/speech-enhancement-asr/data/LibriSpeech/dev-noise-subtractive-250ms-1/84/121123/84-121123-0001.flac'

        input_spectrogram, _, _, _, _ = load_audio_spectrogram(
            corrupted_file_path, normalize_spect=self.normalize)

        mask_filepath = path.splitext(corrupted_file_path)[
            0] + '-mask.npy'
        mask = np.loadtxt(mask_filepath, dtype=np.float32)
        width = input_spectrogram.size(0)
        max_size = (len(mask) // width) * width
        mask_vector = torch.Tensor(mask[:max_size].reshape(width, -1).max(axis=1))
        output_sliced = mask_vector

        if not self.mask:
            clean_file_path = self.clean_file_paths[index]
            # clean_file_path = '/home/shamoon/speech-enhancement-asr/data/LibriSpeech/dev-clean/84/121123/84-121123-0001.flac'

            output_spectrogram, _, _, _, _ = load_audio_spectrogram(
                clean_file_path, normalize_spect=self.normalize)

            masked_output_spectrogram = mask_vector.unsqueeze(1) * output_spectrogram
            masked_output_spectrogram = masked_output_spectrogram[mask_vector != 0]

            output_sliced = masked_output_spectrogram

        return input_spectrogram, output_sliced


def pad_samples(batched_data, padding_value=0):
    (xx, yy) = zip(*batched_data)

    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=padding_value)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=padding_value)

    return xx_pad, yy_pad, x_lens, y_lens


def pad_samples_audio(batched_data):
    return pad_samples(batched_data, padding_value=0)
