import soundfile as sf
import scipy
import librosa
import numpy as np
import warnings
import torch

warnings.filterwarnings("ignore", category=UserWarning)


# def load_mel_spectrogram(audio_path):
#     sample_rate, samples = read(audio_path)
#     samples = samples.astype('float32') / 32767
#     # samples, sample_rate = sf.read(audio_path)

#     # Taken from https://github.com/PaddlePaddle/DeepSpeech/blob/766e96e600795cea4187123b9ed76dcd250f2d04/data_utils/featurizer/audio_featurizer.py#L121
#     n_fft = int(sample_rate * 0.001 * 20)  # 20ms
#     hop_length = n_fft // 4
#     # net_segments = n_fft - hop_length

#     # seconds_per_segment = net_segments / sample_rate
#     # ms_per_segment = int(seconds_per_segment * 1000)
#     # print('ms_per_segment', ms_per_segment)

#     melspectrogram = librosa.feature.melspectrogram(
#         y=samples, sr=sample_rate, window=scipy.signal.hanning, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
#     # print(melspectrogram.shape)

#     # By default, the first axis is frequencies and the second is time.
#     # We swap them here.
#     melspectrogram = np.swapaxes(melspectrogram, 0, 1)
#     log_spectrogram = np.log1p(melspectrogram)

#     return log_spectrogram


def load_audio_spectrogram(audio_path, transpose=True, normalize_spect=False):

    sample_rate = librosa.get_samplerate(audio_path)
    samples, sample_rate = librosa.core.load(audio_path, sr=sample_rate)

    n_fft, hop_length = get_n_fft_overlap(sample_rate)
    # print('n_fft: {}\thop_length: {}\twin_length: {}\twindow: {}'.format(
    #     n_fft, hop_length, n_fft, scipy.signal.hamming))
    # print(samples.shape, samples.mean())
    D = librosa.stft(samples, n_fft=n_fft, hop_length=hop_length,
                     win_length=n_fft, window=scipy.signal.hamming)

    spect, _ = librosa.magphase(D)
    spect = np.log1p(spect)
    # print('After LOG1p: ', spect.shape, spect.mean())

    if transpose:
        spect = np.swapaxes(spect, 0, 1)
    spect = torch.FloatTensor(spect)
    # print('After FloatTensor: ', spect.shape, spect.mean())

    if normalize_spect:
        mean = spect.mean()
        std = spect.std()
        spect.add_(-mean)
        spect.div_(std)
    # print('After normalize: ', spect.shape, spect.mean())

    return spect.contiguous(), len(samples), sample_rate, n_fft, hop_length

# def load_audio_spectrogram_scipy(audio_path):
#     samples, sample_rate = sf.read(audio_path)

#     print('samples.shape', samples.shape)

#     n_fft, hop_length = get_n_fft_overlap(sample_rate)
#     print(n_fft, hop_length)

#     _, _, spectrogram = signal.spectrogram(
#         samples, fs=sample_rate, nperseg=n_fft, noverlap=hop_length, window=signal.hann(n_fft))

#     # By default, the first axis is frequencies and the second is time.
#     # We swap them here.
#     spectrogram = np.swapaxes(spectrogram, 0, 1)
#     log_spectrogram = np.log1p(spectrogram)

#     return log_spectrogram


def load_times_frequencies(audio_path):
    samples, sample_rate = sf.read(audio_path)
    # Taken from https://github.com/PaddlePaddle/DeepSpeech/blob/766e96e600795cea4187123b9ed76dcd250f2d04/data_utils/featurizer/audio_featurizer.py#L121
    n_fft, hop_length = get_n_fft_overlap(sample_rate)

    frequencies, times, _ = scipy.signal.spectrogram(
        samples, fs=sample_rate, nperseg=n_fft, noverlap=hop_length, window=scipy.signal.hann(n_fft))

    return times, frequencies


def create_audio_from_spectrogram(spectrogram, n_fft, hop_length, length):
    spectrogram = np.swapaxes(spectrogram, 0, 1)
    audio_signal = librosa.griffinlim(
        spectrogram, n_iter=128, win_length=n_fft, hop_length=hop_length, window=scipy.signal.hamming, length=length)

    return audio_signal


def get_n_fft_overlap(sample_rate, time_ms=20):
    # Taken from https://github.com/PaddlePaddle/DeepSpeech/blob/766e96e600795cea4187123b9ed76dcd250f2d04/data_utils/featurizer/audio_featurizer.py#L121

    n_fft = int(sample_rate * 0.001 * time_ms)  # 20ms
    hop_length = int(sample_rate * 0.001 * (time_ms // 2))  # 10ms

    return n_fft, hop_length
