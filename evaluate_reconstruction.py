import json
import importlib
import sys
import argparse
import torch
import os
import glob
from pathlib import Path
import soundfile as sf
from utils.audio_util import load_audio_spectrogram, load_times_frequencies, create_audio_from_spectrogram
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path", help="Model to Load", type=str)

    parser.add_argument('--mask_wandb', type=str, help='Run mask based on a wandb id')

    parser.add_argument('--wandb', type=str, help='Run based on a wandb id')

    parser.add_argument(
        "--audio_path", help="Audio file", type=str)

    parser.add_argument(
        "--seq_length", help="Sequence Length", type=int, default=20)

    parser.add_argument('--saved_model_path', help='File of the saved model',
                        default='saved_models')

    args = parser.parse_args()

    return args

def load_masking_model(wandb_id, device):
    wandb_dir = list(glob.iglob(os.path.join('wandb', '*' + wandb_id), recursive=False))[0]
    model_path = os.path.join(wandb_dir, 'best-model.pt')

    (head, tail) = os.path.split(model_path)
    mask_args_path = os.path.join(head, tail.replace('best-model.pt', 'args.json'))
    masked_args = json.loads(open(mask_args_path, 'r').read())

    model = importlib.import_module('masking_model').MaskingModel(
        feature_dim=161, kernel_size=masked_args['kernel_size'], kernel_size_step=masked_args['kernel_size_step'], final_kernel_size=masked_args['final_kernel_size'], device='cpu')

    state_dict = torch.load(model_path, map_location=device)

    model.load_state_dict(state_dict)

    model = model.float()
    model.eval()

    return model

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask_model = load_masking_model(args.mask_wandb, device)

    wandb_dir = list(glob.iglob(os.path.join('wandb', '*' + args.wandb), recursive=False))[0]
    model_path = os.path.join(wandb_dir, 'best-model.pt')
    saved_model_path = wandb_dir
    args_path = os.path.join(wandb_dir, 'args.json')
    saved_args = json.loads(open(args_path, 'r').read())

    sys.path.append(os.path.abspath(saved_model_path))
    model = importlib.import_module('reconstruction_model').ReconstructionModel(
        feature_dim=saved_args['feature_dim'], kernel_size=saved_args['kernel_size'], kernel_size_step=saved_args['kernel_size_step'], final_kernel_size=saved_args['final_kernel_size'])

    state_dict = torch.load(model_path, map_location=device)

    model.load_state_dict(state_dict)

    model = model.float()
    model.eval()

    filename_without_ext = Path(args.audio_path).stem

    input_spectrogram, samples_length, sample_rate, n_fft, hop_length = load_audio_spectrogram(
        args.audio_path)

    input_spectrogram = input_spectrogram.view(
        1, input_spectrogram.size(0), input_spectrogram.size(1))

    print('input_spectrogram\tMean: {:.4f} ± {:.4f}\tMin: {:.4f}\tMax: {:.4f}\tSize: {}'.format(torch.mean(input_spectrogram), torch.std(
        input_spectrogram), torch.min(input_spectrogram), torch.max(input_spectrogram), input_spectrogram.size()))

    mask = mask_model(input_spectrogram)
    mask = torch.round(mask).float()
    expanded_mask = mask_model.expand_mask(mask, seq_length=input_spectrogram.size(1))
    masked_input = input_spectrogram * expanded_mask[..., None]

    # torch.set_printoptions(profile='full', precision=4)
    # print(expanded_mask.size(), input_spectrogram.size(), masked_input.size())
    # print(expanded_mask[0], expanded_mask[0].size())
    # print(input_spectrogram[0,:,0], input_spectrogram[0,:,0].size())
    # print(masked_input[0,:,0], masked_input[0,:,0].size())
    # quit()

    # Model takes data of shape: torch.Size([BATCH_SIZE, SEQUENCE_LENGTH, FEATURE_DIM])
    output = model(masked_input)

    output[mask == 0] = input_spectrogram[mask == 0] 

    print('model output\t\tMean: {:.4f} ± {:.4f}\tMin: {:.4f}\tMax: {:.4f}\tSize: {}'.format(
        torch.mean(output), torch.std(output), torch.min(output), torch.max(output), output.size()))

    output = torch.expm1(output)
    print('expm1 output\t\tMean: {:.4f} ± {:.4f}\tMin: {:.4f}\tMax: {:.4f}\tSize: {}'.format(torch.mean(output), torch.std(output), torch.min(output), torch.max(output), output.size()))

    np_output = output.view(output.size(1), output.size(2)).detach().numpy()

    # output requires shape of [SEQUENCE_LEN, FEATURE_DIM]
    audio = create_audio_from_spectrogram(
        np_output, n_fft=n_fft, hop_length=hop_length, length=samples_length)

    sf.write(filename_without_ext + '.wav', audio, sample_rate)

if __name__ == '__main__':
    main()
