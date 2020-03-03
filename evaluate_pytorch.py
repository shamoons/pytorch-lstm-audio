import matplotlib.pyplot as plt
import argparse
import torch
import soundfile as sf
from audio_util import load_audio_spectrogram, load_times_frequencies, create_audio_from_spectrogram
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path", help="Model to Load", type=str)

    parser.add_argument(
        "--audio_path", help="Audio file", type=str)

    parser.add_argument(
        "--seq_length", help="Sequence Length", type=int, default=5)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    model = torch.load(args.model_path)
    model = model.float()
    model.eval()

    input_spectrogram = load_audio_spectrogram(args.audio_path)
    print('input_spectrogram.shape', input_spectrogram.shape)

    # times, frequencies = load_times_frequencies(args.audio_path)

    timesteps = input_spectrogram.shape[0]

    remainder = args.seq_length - timesteps % args.seq_length
    batches = (timesteps + remainder) // args.seq_length

    reshaped_input_spectrogram = np.append(input_spectrogram, np.zeros(
        (remainder, 161)), axis=0)

    reshaped_input_spectrogram = reshaped_input_spectrogram.reshape(
        (batches, args.seq_length, 161))

    # reshaped_input_spectrogram = np.swapaxes(reshaped_input_spectrogram, 0, 1)

    tensor = torch.from_numpy(reshaped_input_spectrogram).float()

    # print(input_spectrogram)
    output, _ = model(tensor)
    np_output = output.detach().numpy()

    output = np_output.reshape(timesteps + remainder, 161)

    output = output[: -remainder, :]

    audio = create_audio_from_spectrogram(
        output, n_fft=320, hop_length=80)

    # samples, sample_rate = sf.read(args.audio_path)
    sf.write('test.wav', audio, 16000)


if __name__ == '__main__':
    main()
