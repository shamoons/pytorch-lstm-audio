
import matplotlib.pyplot as plt
from scipy import signal
import soundfile as sf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--file", help="File", type=str)
args = parser.parse_args()

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
