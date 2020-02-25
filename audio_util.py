import soundfile as sf
import scipy
from scipy import signal
import librosa
import numpy as np


def load_audio_spectrogram(audio_path, n_mels=128, eps=1e-10):
    samples, sample_rate = sf.read(audio_path)
    # Taken from https://github.com/PaddlePaddle/DeepSpeech/blob/766e96e600795cea4187123b9ed76dcd250f2d04/data_utils/featurizer/audio_featurizer.py#L121

    n_fft = int(sample_rate * 0.001 * 20)  # 20ms
    hop_length = n_fft // 4

    # melspectrogram = librosa.feature.melspectrogram(
    #     y=samples, sr=sample_rate, window=scipy.signal.hanning, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    # print(melspectrogram.shape)

    _, _, spectrogram = signal.spectrogram(
        samples, fs=sample_rate, nperseg=n_fft, noverlap=hop_length, window=signal.hann(n_fft))

    # By default, the first axis is frequencies and the second is time.
    # We swap them here.
    spectrogram = np.swapaxes(spectrogram, 0, 1)
    log_spectrogram = np.log(spectrogram + eps)

    return log_spectrogram


def load_times_frequencies(audio_path):
    samples, sample_rate = sf.read(audio_path)
    # Taken from https://github.com/PaddlePaddle/DeepSpeech/blob/766e96e600795cea4187123b9ed76dcd250f2d04/data_utils/featurizer/audio_featurizer.py#L121
    n_fft = int(sample_rate * 0.001 * 20)  # 20ms
    hop_length = n_fft // 4

    frequencies, times, _ = signal.spectrogram(
        samples, fs=sample_rate, nperseg=nperseg, noverlap=hop_length, window=signal.hann(n_fft))

    return times, frequencies
