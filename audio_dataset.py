import glob
import random
import torch
import numpy as np
import os.path as path
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset
from utils.audio_util import load_audio_spectrogram


class AudioDataset(Dataset):
    def __init__(self, corrupted_path, seq_length=10, feature_dim=5, train_set=False, test_set=False, normalize=False, repeat_sample=1):
        torch.manual_seed(0)
        np.random.seed(0)
        self.seq_length = seq_length
        self.feature_dim = feature_dim
        self.normalize = normalize

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

        # self.clean_file_paths = self.clean_file_paths[136:138]
        # self.corrupted_file_paths = self.corrupted_file_paths[136:138]

        self.clean_file_paths = np.repeat(self.clean_file_paths, repeat_sample)
        self.corrupted_file_paths = np.repeat(
            self.corrupted_file_paths, repeat_sample)

    def __len__(self):
        return len(self.clean_file_paths)

    def __getitem__(self, index):
        input_sliced = []
        output_sliced = []

        input_spectrogram, _, _, _, _ = load_audio_spectrogram(
            self.corrupted_file_paths[index], normalize_spect=self.normalize)

        output_spectrogram, _, _, _, _ = load_audio_spectrogram(
            self.clean_file_paths[index], normalize_spect=self.normalize)

        averaged_time_energy_input = torch.mean(input_spectrogram, dim=1)
        soft_min_inputs = torch.nn.Softmin()(averaged_time_energy_input).detach().numpy()
        input_indices = np.arange(0, input_spectrogram.size(0))
        mid_index = np.random.choice(input_indices, p=soft_min_inputs)

        if mid_index < self.seq_length // 2:
            mid_index = self.seq_length // 2
        if mid_index > (len(input_spectrogram) - self.seq_length) // 2:
            mid_index = (len(input_spectrogram) - self.seq_length) // 2

        start_index = max(mid_index - self.seq_length // 2, 0)
        end_index = start_index + self.seq_length

        input_sliced = input_spectrogram[start_index:end_index]
        output_sliced = output_spectrogram[start_index:end_index]

        inputs_array = np.array(input_sliced, dtype=np.float32)
        outputs_array = np.array(output_sliced, dtype=np.float32)

        return inputs_array, outputs_array

    def __getitem__RAND(self, index):
        random_tensor = torch.rand(self.seq_length, self.feature_dim) * 2
        random_tensor = random_tensor - 0.5

        return random_tensor, random_tensor * 2
