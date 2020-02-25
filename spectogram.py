import matplotlib.pyplot as plt
from scipy import signal
import soundfile as sf
import argparse
import scipy
import librosa
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
        print('max: ', max_spectrogram, '\tCurrent: ',
              current_max, '\tmean:', np.mean(spectrogram), '\tstd: ', np.std(spectrogram))

else:
    samples, sample_rate = sf.read(args.file)
    print(samples, samples.shape)
    print('sample_rate: ', sample_rate)
    time_per_segment_ms = 20
    nperseg = int(sample_rate * 0.001 * time_per_segment_ms)
    # overlap = nperseg // 4
    # https://stackoverflow.com/questions/46635958/python-scipy-how-to-set-the-time-frame-for-a-spectrogram
    overlap = int(
        sample_rate * (time_per_segment_ms - time_per_segment_ms)/1000)

    print(nperseg, 'nperseg')
    print(overlap, 'overlap')
    frequencies, times, spectrogram = signal.spectrogram(
        samples, sample_rate, nperseg=nperseg, window=signal.hann(nperseg), noverlap=overlap, mode='magnitude')

    print(times)
    print('times.shape', times.shape)
    print('spectrogram.shape', spectrogram.shape)

    # plt.pcolormesh(times, frequencies, spectrogram)
    # plt.imshow(spectrogram)
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()

    audio_signal = librosa.griffinlim(
        spectrogram, n_iter=128, win_length=nperseg, hop_length=overlap, window=signal.hann(nperseg))
    print(audio_signal, audio_signal.shape)

    sf.write('test.wav', audio_signal, sample_rate)
