import glob
import numpy as np
from audio_util import load_audio_spectrogram


def main():
    clean_audio_file_paths = np.array(
        sorted(glob.iglob('data/**/*.flac', recursive=True)))

    # spects = np.empty((0, 0, 161))

    for filepath in clean_audio_file_paths:
        spect, _, _, _, _ = load_audio_spectrogram(filepath)
        spect = spect.data.numpy()
        # spects.append(spect)
        print(spect.shape)
    # print(len(clean_audio_file_paths))


if __name__ == '__main__':
    main()
