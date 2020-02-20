import matplotlib.pyplot as plt
from scipy import signal
import soundfile as sf
import argparse
import glob
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "--file", help="File", type=str)
parser.add_argument(
    '--get_max', help='Whether or not to run through all files to get max spectogram value', type=bool)
args = parser.parse_args()


if args.get_max == True:
    current_max = 0
    for file_path in glob.iglob('data//**/*.flac', recursive=True):
        samples, sample_rate = sf.read(file_path)
        nperseg = int(sample_rate * 0.001 * 20)
        frequencies, times, spectrogram = signal.spectrogram(
            samples, sample_rate, nperseg=nperseg, window=signal.hann(nperseg))
        max_spectrogram = np.max(spectrogram)
        if max_spectrogram > current_max:
            current_max = max_spectrogram
        print('max: ', max_spectrogram, '\tCurrent: ', current_max)

else:
    samples, sample_rate = sf.read(args.file)
    nperseg = int(sample_rate * 0.001 * 20)
    frequencies, times, spectrogram = signal.spectrogram(
        samples, sample_rate, nperseg=nperseg, window=signal.hann(nperseg))

    print(len(times))

    plt.pcolormesh(times, frequencies, spectrogram)
    plt.imshow(spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
