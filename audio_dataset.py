import glob
import random
import torch
import math
import numpy as np
import os.path as path
from torch.utils.data import Dataset
from utils.audio_util import load_audio_spectrogram


class AudioDataset(Dataset):
    def __init__(self, corrupted_path, mask, feature_dim=5, train_set=False, test_set=False, normalize=False, repeat_sample=1, batch_size=64, shuffle=True):
        torch.manual_seed(0)

        self.feature_dim = feature_dim
        self.normalize = normalize
        self.batch_size = batch_size
        self.shuffle = shuffle
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

        cutoff_index = int(len(corrupted_audio_file_paths) * 0.9)

        if train_set:
            if not self.mask:
                self.clean_file_paths = clean_audio_file_paths[0: cutoff_index]
            self.corrupted_file_paths = corrupted_audio_file_paths[0: cutoff_index]
        if test_set:
            if not self.mask:
                self.clean_file_paths = clean_audio_file_paths[cutoff_index:]
            self.corrupted_file_paths = corrupted_audio_file_paths[cutoff_index:]

        if not self.mask:
            self.clean_file_paths = np.repeat(self.clean_file_paths, repeat_sample)
        self.corrupted_file_paths = np.repeat(
            self.corrupted_file_paths, repeat_sample)


    def __len__(self):
        return len(self.corrupted_file_paths) // self.batch_size

    def __getitem__(self, index):
        # if self.shuffle:
        #     np.random.seed(index)
        #     index = np.random.choice(indices)

        # if index > len(self.corrupted_file_paths):
        #     continue

        corrupted_file_path = self.corrupted_file_paths[index]

        corrupted_file_path = '/home/shamoon/speech-enhancement-asr/data/LibriSpeech/dev-noise-subtractive-250ms-1/84/121123/84-121123-0000.flac'

        input_spectrogram, _, _, _, _ = load_audio_spectrogram(
            corrupted_file_path, normalize_spect=self.normalize)

        if not self.mask:
            clean_file_path = self.clean_file_paths[index]
            clean_file_path = '/home/shamoon/speech-enhancement-asr/data/LibriSpeech/dev-clean/84/121123/84-121123-0000.flac'
            output_spectrogram, _, _, _, _ = load_audio_spectrogram(
                clean_file_path, normalize_spect=self.normalize)
        else:
            mask_filepath = path.splitext(corrupted_file_path)[
                0] + '-mask.npy'
            mask = np.loadtxt(mask_filepath, dtype=np.float32)

            width = input_spectrogram.size(0)
            max_size = (len(mask) // width) * width
            mask_vector = mask[:max_size].reshape(width, -1).max(axis=1)

        # if seq_length >= input_spectrogram.size(0):
        #     input_spectrogram = input_spectrogram.repeat(
        #         1 + (seq_length // input_spectrogram.size(0)), 1)
            
        #     if self.mask:
        #         mask_vector = mask_vector.repeat(
        #             1 + (seq_length // len(mask_vector)), 0)
        #     else:
        #         output_spectrogram = output_spectrogram.repeat(
        #             1 + (seq_length // output_spectrogram.size(0)), 1)

        # averaged_time_energy_input = torch.mean(input_spectrogram, dim=1)
        # soft_min_inputs = torch.nn.Softmin()(averaged_time_energy_input).detach().numpy()
        # input_indices = np.arange(0, input_spectrogram.size(0))
        # mid_index = np.random.choice(input_indices, p=soft_min_inputs)

        # mid_index = input_spectrogram.size(0) // 2 #TODO: Remove this hardcoding

        # if mid_index < seq_length // 2:
        #     mid_index = seq_length // 2
        # if mid_index > (input_spectrogram.size(0) - seq_length) // 2:
        #     mid_index = (input_spectrogram.size(0) - seq_length) // 2

        # start_index = max(mid_index - seq_length // 2, 0)
        # end_index = start_index + seq_length

        # input_sliced = input_spectrogram[start_index:end_index].numpy()
        input_sliced = input_spectrogram.numpy()
        if self.mask:
            output_sliced = mask_vector
        else:
            output_sliced = output_spectrogram.numpy()
        
        input_sliced = np.array(input_sliced, dtype=np.float32)
        output_sliced = np.array(output_sliced, dtype=np.float32)
        return input_sliced, output_sliced

        # batched_inputs.append(input_sliced)
        # batched_outputs.append(output_sliced)

        # batched_inputs = np.array(batched_inputs, dtype=np.float32)
        # batched_outputs = np.array(batched_outputs, dtype=np.float32)

        # batched_inputs = torch.from_numpy(batched_inputs)
        # batched_outputs = torch.from_numpy(batched_outputs)

        # return batched_inputs, batched_outputs
