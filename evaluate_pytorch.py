import json
import importlib
import sys
import argparse
import torch
import os
from pathlib import Path
import soundfile as sf
from utils.audio_util import load_audio_spectrogram, load_times_frequencies, create_audio_from_spectrogram
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path", help="Model to Load", type=str)

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

    model = torch.load(args.model_path, map_location=torch.device('cpu'))

    saved_args = json.loads(open('saved_models/args.json', 'r').read())
    args.seq_length = saved_args['seq_length']

    sys.path.append(os.path.abspath(args.saved_model_path))
    model = importlib.import_module(
        'saved_model').BaselineModel(feature_dim=saved_args['feature_dim'],
                                     hidden_size=saved_args['hidden_size'], seq_length=saved_args['seq_length'], num_layers=saved_args['num_layers'])
    state_dict = torch.load(args.model_path, map_location='cpu')

    model.load_state_dict(state_dict)

    model = model.float()
    model.eval()

    filename_without_ext = Path(args.audio_path).stem

    input_spectrogram, samples_length, sample_rate, n_fft, hop_length = load_audio_spectrogram(
        args.audio_path)
    print('input_spectrogram', torch.mean(input_spectrogram), torch.std(input_spectrogram), torch.min(
        input_spectrogram), torch.max(input_spectrogram))

    timesteps = input_spectrogram.shape[0]

    remainder = args.seq_length - timesteps % args.seq_length
    batches = (timesteps + remainder) // args.seq_length

    reshaped_input_spectrogram = np.append(input_spectrogram, np.zeros(
        (remainder, 161)), axis=0)

    reshaped_input_spectrogram = reshaped_input_spectrogram.reshape(
        (batches, args.seq_length, 161))

    tensor = torch.from_numpy(reshaped_input_spectrogram).float()

    output, _ = model(tensor)
    np_output = output.detach().numpy()

    output = np_output.reshape(timesteps + remainder, 161)

    output = output[: -remainder, :]

    print('output', np.mean(output), np.std(
        output), np.min(output), np.max(output))
    output = np.expm1(output)
    print('expm1 output', np.mean(output), np.min(output), np.max(output))

    audio = create_audio_from_spectrogram(
        output, n_fft=n_fft, hop_length=hop_length, length=samples_length)

    sf.write(filename_without_ext + '.wav', audio, sample_rate)


if __name__ == '__main__':
    main()
