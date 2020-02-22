import matplotlib.pyplot as plt
import argparse
from audio_util import load_audio_spectrogram, load_times_frequencies
from tensorflow.keras.models import load_model
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path", help="Model to Load", type=str)

    parser.add_argument(
        "--audio_path", help="Audio file", type=str)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    model = load_model(args.model_path)
    print(model.summary())

    input_spectrogram = load_audio_spectrogram(args.audio_path)
    print('input_spectrogram.shape', input_spectrogram.shape)

    start_index = 0
    end_index = start_index + 100

    output_spectrogram = []

    times, frequencies = load_times_frequencies(args.audio_path)

    # while end_index <= len(input_spectrogram):
    while end_index <= start_index + 100:
        input_spectrogram_slice = np.array(
            [input_spectrogram[start_index:end_index]])
        print('input', input_spectrogram_slice.shape)
        output = model.predict(input_spectrogram_slice)

        output_spectrogram.append(output[0])

        start_index = end_index + 1
        end_index = start_index + 100

    print(np.array(output_spectrogram).shape)

    plt.pcolormesh(times, frequencies, output_spectrogram)
    plt.imshow(output_spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

    return


if __name__ == '__main__':
    main()
