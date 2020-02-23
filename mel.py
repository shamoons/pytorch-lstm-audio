import argparse
import numpy as np
import soundfile as sf
import librosa
import librosa.display
import scipy
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file", help="File", type=str)
    args = parser.parse_args()

    samples, sample_rate = sf.read(args.file)
    print(samples, samples.shape)

    n_fft = int(sample_rate * 0.001 * 20)
    hop_length = n_fft // 4

    # nperseg = int(round(window_size * sample_rate / 1e3))
    # noverlap = int(round(step_size * sample_rate / 1e3))

    melspectrogram = librosa.feature.melspectrogram(
        y=samples, sr=sample_rate, window=scipy.signal.hanning, n_fft=n_fft, hop_length=hop_length, n_mels=256)
    log_melspectrogram = np.log(melspectrogram + 1e-10)
    normalized_melspectrogram = (
        log_melspectrogram - log_melspectrogram.mean()) / log_melspectrogram.std()

    print('melspectrogram.shape', melspectrogram.shape)
    print(melspectrogram)
    print(log_melspectrogram)
    print(normalized_melspectrogram)

    # log_S = librosa.power_to_db(melspectrogram, ref=np.max)
    # # Make a new figure
    # plt.figure(figsize=(12, 4))
    # # Display the spectrogram on a mel scale
    # # sample rate and hop length parameters are used to render the time axis
    # librosa.display.specshow(log_S, sr=sample_rate,
    #                          x_axis='time', y_axis='mel', hop_length=hop_length)
    # # Put a descriptive title on the plot
    # plt.title('mel power spectrogram')
    # # draw a color bar
    # plt.colorbar(format='%+02.0f dB')
    # # Make the figure layout compact
    # plt.tight_layout()
    # plt.tight_layout()
    # plt.show()

    audio_signal = librosa.feature.inverse.mel_to_audio(
        melspectrogram, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, window=scipy.signal.hanning, n_iter=256)
    print(audio_signal, audio_signal.shape)

    sf.write('test.wav', audio_signal, sample_rate)


if __name__ == '__main__':
    main()
