import soundfile as sf
from scipy import signal
import numpy as np


def load_audio_spectrogram(audio_path):
    samples, sample_rate = sf.read(audio_path)
    # Taken from https://github.com/PaddlePaddle/DeepSpeech/blob/766e96e600795cea4187123b9ed76dcd250f2d04/data_utils/featurizer/audio_featurizer.py#L121
    nperseg = int(sample_rate * 0.001 * 20)  # 20ms

    _, _, spectrogram = signal.spectrogram(
        samples, fs=sample_rate, nperseg=nperseg, noverlap=nperseg // 2, window=signal.hann(nperseg))

    # By default, the first axis is frequencies and the second is time.
    # We swap them here.
    spectrogram = np.swapaxes(spectrogram, 0, 1)

    return spectrogram


def load_times_frequencies(audio_path):
    samples, sample_rate = sf.read(audio_path)
    # Taken from https://github.com/PaddlePaddle/DeepSpeech/blob/766e96e600795cea4187123b9ed76dcd250f2d04/data_utils/featurizer/audio_featurizer.py#L121
    nperseg = int(sample_rate * 0.001 * 20)  # 20ms

    frequencies, times, _ = signal.spectrogram(
        samples, fs=sample_rate, nperseg=nperseg, noverlap=nperseg // 2, window=signal.hann(nperseg))

    return frequencies, times
