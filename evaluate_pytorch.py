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

    parser.add_argument('--wandb', type=str, help='Run based on a wandb id')

    parser.add_argument(
        "--audio_path", help="Audio file", type=str)

    parser.add_argument(
        "--seq_length", help="Sequence Length", type=int, default=20)

    parser.add_argument('--saved_model_path', help='File of the saved model',
                        default='saved_models')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.wandb:
        wandb_dir = list(glob.iglob(os.path.join(
            'wandb', '*' + args.wandb), recursive=False))[0]
        model_path = os.path.join(wandb_dir, 'best-model.pt')
        saved_model_path = wandb_dir
    else:
        model_path = args.model_path
        saved_model_path = args.saved_model_path

    model = torch.load(model_path, map_location=device)


    sys.path.append(os.path.abspath(saved_model_path))
    model = importlib.import_module('saved_model').BaselineModel(
        feature_dim=161)

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

    # Model takes data of shape: torch.Size([BATCH_SIZE, SEQUENCE_LENGTH, FEATURE_DIM])
    output = model(input_spectrogram)

    print('model output\t\tMean: {:.4f} ± {:.4f}\tMin: {:.4f}\tMax: {:.4f}\tSize: {}'.format(
        torch.mean(output), torch.std(output), torch.min(output), torch.max(output), output.size()))

    output = torch.expm1(output)
    print('expm1 output\t\tMean: {:.4f} ± {:.4f}\tMin: {:.4f}\tMax: {:.4f}\tSize: {}'.format(torch.mean(output), torch.std(output), torch.min(output), torch.max(output), output.size()))

    np_output = output.view(output.size(1), output.size(2)).detach().numpy()

    # output requires shape of [SEQUENCE_LEN, FEATURE_DIM]
    audio = create_audio_from_spectrogram(
        np_output, n_fft=n_fft, hop_length=hop_length, length=samples_length)

    sf.write(filename_without_ext + '.wav', audio, sample_rate)

    quit()

    timesteps = input_spectrogram.shape[0]

    remainder = args.seq_length - timesteps % args.seq_length
    batches = (timesteps + remainder) // args.seq_length

    reshaped_input_spectrogram = np.append(input_spectrogram, np.zeros(
        (remainder, 161)), axis=0)

    reshaped_input_spectrogram = reshaped_input_spectrogram.reshape(
        (batches, args.seq_length, 161))

    tensor = torch.from_numpy(reshaped_input_spectrogram).float()

    output = model(tensor)
    np_output = output.detach().numpy()

    output = np_output.reshape(timesteps + remainder, 161)

    output = output[: -remainder, :]

    output = np.expm1(output)
    print('expm1 output\t\tMean: {:.4f}\tSTD: {:.4f}\tMin: {:.4f}\tMax: {:.4f}\tSize: {}'.format(
        np.mean(output), np.std(output), np.min(output), np.max(output), output.shape))

    audio = create_audio_from_spectrogram(
        output, n_fft=n_fft, hop_length=hop_length, length=samples_length)

    sf.write(filename_without_ext + '.wav', audio, sample_rate)


if __name__ == '__main__':
    main()
