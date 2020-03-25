import glob
import random
import torch
import math
import numpy as np
import os.path as path
from torch.utils.data import Dataset
from utils.audio_util import load_audio_spectrogram


class AudioDataset(Dataset):
    def __init__(self, corrupted_path, seq_length=[8], feature_dim=5, train_set=False, test_set=False, normalize=False, repeat_sample=1, batch_size=64, shuffle=True):
        torch.manual_seed(0)

        # If the seq_length is a list, then choose a random sequence
        if len(seq_length) > 1:
            self.seq_length = seq_length
        else:
            self.seq_length = seq_length[0]

        self.feature_dim = feature_dim
        self.normalize = normalize
        self.batch_size = batch_size
        self.shuffle = shuffle

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

        # clean_audio_file_paths = np.array(
        #     sorted(glob.iglob(clean_base_path + '/**/*.flac', recursive=True)))


        cutoff_index = int(len(corrupted_audio_file_paths) * 0.9)

        if train_set:
            # self.clean_file_paths = clean_audio_file_paths[0: cutoff_index]
            self.corrupted_file_paths = corrupted_audio_file_paths[0: cutoff_index]
        if test_set:
            # self.clean_file_paths = clean_audio_file_paths[cutoff_index:]
            self.corrupted_file_paths = corrupted_audio_file_paths[cutoff_index:]


        # self.clean_file_paths = np.repeat(self.clean_file_paths, repeat_sample)
        self.corrupted_file_paths = np.repeat(
            self.corrupted_file_paths, repeat_sample)

    def __len__(self):
        return len(self.corrupted_file_paths) // self.batch_size

    def __getitem__(self, index):
        if isinstance(self.seq_length, list):
            seq_length = np.random.randint(self.seq_length[0], self.seq_length[1] + 1)
        else:
            seq_length = self.seq_length

        batched_inputs = []
        batched_outputs = []

        indices = np.arange(0, len(self.corrupted_file_paths))
        while len(batched_inputs) < self.batch_size:
            if self.shuffle:
                np.random.seed(index)
                index = np.random.choice(indices)

            if index > len(self.corrupted_file_paths):
                continue
            input_spectrogram, _, sample_rate, n_fft, hop_length = load_audio_spectrogram(
                self.corrupted_file_paths[index], normalize_spect=self.normalize)
            sec_per_segment = (n_fft - hop_length) / sample_rate
        
            mask_filepath = path.splitext(self.corrupted_file_paths[index])[0] +'-mask.npy'
            mask = np.loadtxt(mask_filepath, dtype=np.float32)
            width = input_spectrogram.size(0)
            # width = len(mask) / sample_rate
            # width = math.ceil(width * 100)
            max_size = (len(mask)//width) * width
            mask_vector = mask[:max_size].reshape(width, -1).max(axis=1)
            # print(mask.shape, mask.shape[0] // input_spectrogram.size(0))
            mask_seconds = len(mask) / sample_rate
            # print(mask.shape, mask_vector.shape)
            # print(f"mask_filepath: {mask_filepath}\tfile: {self.corrupted_file_paths[index]}\tmask_len: {len(mask)}\tmask_seconds: {mask_seconds}\tinp_spect: {input_spectrogram.size()}\tsec_per_segment: {sec_per_segment}")
            # quit()
            # print(self.corrupted_file_paths[index], mask.mean(), len(mask), len(input_spectrogram))

            # output_spectrogram, _, _, _, _ = load_audio_spectrogram(
            #     self.clean_file_paths[index], normalize_spect=self.normalize)
            
            if seq_length >= input_spectrogram.size(0):
                input_spectrogram = input_spectrogram.repeat(1 + (seq_length // input_spectrogram.size(0)), 1)
                # output_spectrogram = output_spectrogram.repeat(1 + (seq_length // output_spectrogram.size(0)), 1)

            averaged_time_energy_input = torch.mean(input_spectrogram, dim=1)
            soft_min_inputs = torch.nn.Softmin()(averaged_time_energy_input).detach().numpy()
            input_indices = np.arange(0, input_spectrogram.size(0))
            mid_index = np.random.choice(input_indices, p=soft_min_inputs)


            if mid_index < seq_length // 2:
                mid_index = seq_length // 2
            if mid_index > (input_spectrogram.size(0) - seq_length) // 2:
                mid_index = (input_spectrogram.size(0) - seq_length) // 2

            start_index = max(mid_index - seq_length // 2, 0)
            end_index = start_index + seq_length

            input_sliced = input_spectrogram[start_index:end_index]
            output_sliced = mask_vector[start_index:end_index]

            input_sliced = input_sliced.numpy()

            # print('batched_inputs.shape', batched_inputs.shape)
            batched_inputs.append(input_sliced)
            batched_outputs.append(output_sliced)
            # print('batched_inputs.shape', batched_inputs)
            # batched_outputs = np.append(batched_outputs, output_sliced, axis=0)

        # print('batched_inputs', torch.FloatTensor(batched_inputs).size())
        # print('batched_inputs', np.asarray(batched_inputs).shape)
        # print('batched_outputs', np.asarray(batched_outputs).shape)
        # batched_inputs = torch.stack(batched_inputs)
        batched_inputs = np.asarray(batched_inputs)
        batched_outputs = np.asarray(batched_outputs)
        if batched_outputs.dtype == 'object':
            print(batched_outputs)
        print(batched_outputs.dtype)
        batched_inputs = torch.from_numpy(batched_inputs)
        batched_outputs = torch.from_numpy(batched_outputs)
        # print('seq_length', seq_length)
        # print('batched_inputs', batched_inputs.size())
        # print('batched_outputs', batched_outputs.size())

        # quit()


        return batched_inputs, batched_outputs
