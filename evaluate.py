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

    # start_index = 0
    # end_index = start_index + 100

    # output_spectrogram = np.empty((0, 161))

    times, frequencies = load_times_frequencies(args.audio_path)

    remainder = 100 - input_spectrogram.shape[0] % 100
    print('remainder', remainder)

    reshaped_input_spectrogram = np.append(input_spectrogram, np.zeros(
        (remainder, 161)), axis=0)

    reshaped_input_spectrogram = reshaped_input_spectrogram.reshape(
        (6, 100, 161))

    print('RE input_spectrogram.shape', reshaped_input_spectrogram.shape)
    output = model.predict(reshaped_input_spectrogram)
    output = output.reshape(600, 161)
    output = output[: -remainder, :]

    print('output.shape', output.shape)

    # while end_index < len(input_spectrogram):
    #     print('\n')
    #     # while end_index <= start_index + 100:
    #     input_spectrogram_slice = np.zeros((1, 100, 161))
    #     slice_length = np.array(
    #         [input_spectrogram[start_index:end_index]]).shape[1]
    #     print('slice_length', slice_length)
    #     input_spectrogram_slice[:, : slice_length, :] = np.array(
    #         [input_spectrogram[start_index:end_index]])
    #     print('input.shape', input_spectrogram_slice.shape)
    #     output = model.predict(input_spectrogram_slice)
    #     output_slice = output[0][0:slice_length]
    #     # output_slice = output[0]
    #     print('output.shape', output.shape)

    #     print('output_slice.shape', output_slice.shape)
    #     output_spectrogram = np.concatenate(
    #         (output_spectrogram, output_slice), axis=0)
    #     print('output_spectrogram.shape', output_spectrogram.shape)

    #     start_index = end_index
    #     end_index = min(start_index + 100, len(input_spectrogram))

    #     print('end_index', end_index)

    print('\ntimes.shape', times.shape)
    print('frequencies.shape', frequencies.shape)
    # print('output_spectrogram.shape', np.array(output_spectrogram).shape)

    output_spectrogram = np.swapaxes(input_spectrogram, 0, 1)

    plt.pcolormesh(times, frequencies, output_spectrogram)
    plt.imshow(output_spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

    return


if __name__ == '__main__':
    main()
