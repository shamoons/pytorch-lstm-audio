import glob
import torchaudio
import torch
import numpy
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self, corrupted_path, train_set=False, test_set=False):
        torch.manual_seed(0)
        numpy.random.seed(0)

        audio_file_paths = list(
            sorted(glob.iglob(corrupted_path + '**/*.flac', recursive=True)))

        cutoff_index = int(len(audio_file_paths) * 0.9)

        if train_set == True:
            self.file_paths = audio_file_paths[0: cutoff_index]
        if test_set == True:
            self.file_paths = audio_file_paths[cutoff_index:]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        # print('self.file_paths', len(self.file_paths))

        # corrupted_signal = torchaudio.load(
        #     self.file_paths[index], out=None, normalization=True)
        # corrupted_sound_data = corrupted_signal[0].permute(1, 0)

        # clean_signal = torchaudio.load(
        #     self.file_paths[index], out=None, normalization=True)
        # clean_sound_data = clean_signal[0].permute(1, 0)

        # corrupted_sound_data = corrupted_sound_data[0:100]
        # clean_sound_data = clean_sound_data[0:100]
        # print(corrupted_sound_data.size(), clean_sound_data.size(), '\n')
        random_tensor = torch.rand(4000) * 2
        random_tensor = random_tensor - 1

        return random_tensor, random_tensor * 2
