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

        cutoff_index = int(len(audio_file_paths) * 0.01)

        if train_set == True:
            self.file_paths = audio_file_paths[0: cutoff_index]
        if test_set == True:
            self.file_paths = audio_file_paths[cutoff_index:]
        print('len', len(self.file_paths))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        random_tensor = torch.rand(1, 5) * 2
        random_tensor = random_tensor - 1

        return random_tensor, random_tensor * 2
