import matplotlib.pyplot as plt
import argparse
from audio_util import load_audio_spectrogram, load_times_frequencies
from keras.models import load_model
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

    start_index = 0
    end_index = start_index + 100

    output_spectrogram = []

    times, frequencies = load_times_frequencies(args.audio_path)

    while end_index <= len(input_spectrogram):
        output = model.predict(
            np.array([input_spectrogram[start_index:end_index]]))

        output_spectrogram.append(output[0])

        start_index = end_index + 1
        end_index = start_index + 100

    # print(input_spectrogram[current_start_index].shape)

    plt.pcolormesh(times, frequencies, output_spectrogram)
    plt.imshow(output_spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

    return


if __name__ == '__main__':
    main()
