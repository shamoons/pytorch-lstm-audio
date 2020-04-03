import matplotlib.pyplot as plt
import numpy as np
import librosa
import argparse
import os
import sys
import torch
parentdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parentdir)
from utils.audio_util import load_audio_spectrogram
from utils.model_loader import load_masking_model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--audio_path', help='Path for audio', required=True)
    parser.add_argument('--mask_wandb', help='Weights and bias model for masking', required=True)

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    sample_rate = librosa.get_samplerate(args.audio_path)
    samples, sample_rate = librosa.core.load(args.audio_path, sr=sample_rate)
    spect, _, _, _, _ = load_audio_spectrogram(args.audio_path)

    spect = spect.view(
        1, spect.size(0), spect.size(1))
    print(spect.size())

    # print(samples)
    # print(sample_rate)
    signal_time = np.linspace(0, len(samples) / sample_rate, num=len(samples))


    plt.figure(0)
    # plt.title("Raw Audio Signal")
    plt.plot(signal_time, samples)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')

    plt.axis("off")   # turns off axes
    plt.axis("tight")  # gets rid of white border
    plt.axis("image") 
    plt.tight_layout(pad=0)
    plt.savefig('vizoutputs/audio_signal.png', bbox_inches='tight', pad=0)

    plt.figure(1)
    # plt.title("Spectrogram")
    powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(samples, Fs=sample_rate, cmap='Blues')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Frequency (Hz)')
    
    plt.savefig('vizoutputs/spectrogram.png', bbox_inches='tight', pad=0)

    mask_model = load_masking_model(args.mask_wandb, 'cpu')
    mask_output = torch.nn.Sigmoid()(mask_model(spect))
    mask = torch.round(mask_output[0]).detach().numpy()
    print(mask)

    plt.figure(2)
    x = np.random.randint(0, 2, 31)
    plt.imshow(x.reshape(1, -1), extent=[0, len(x), 0, 1], cmap='Greys')
    plt.xticks(np.arange(0, len(x), 1), [])
    plt.yticks([])
    plt.grid(True, axis='x', lw=1, c='black')
    plt.tick_params(axis='x', length=0)
    # plt.imshow(mask.reshape(1, -1), extent=[0, len(mask), 0, 1], cmap='Greys')
    # plt.xticks(np.arange(0, len(mask), 1), [])
    # plt.yticks([])
    # plt.grid(True, axis='x', lw=1, c='black')
    # plt.tick_params(axis='x', length=0)
    plt.savefig('vizoutputs/reconstruction_mask.png', bbox_inches='tight', pad=0)

    # print(powerSpectrum)

    plt.figure(3)
    spect_exp = torch.expm1(spect).detach().numpy()
    print(spect_exp.flatten().min(), spect_exp.flatten().mean(), spect_exp.flatten().max())
    plt.hist(spect_exp.flatten(), bins=256, range=(0, 2))
    plt.savefig('vizoutputs/raw_spect_hist.png', bbox_inches='tight', pad=0)

    plt.figure(4)
    log_spect = np.log1p(spect_exp.flatten())
    print(log_spect.min(), log_spect.mean(), log_spect.max())
    plt.hist(log_spect, bins=512, range=(0, 4))
    plt.savefig('vizoutputs/raw_spect_hist_log.png', bbox_inches='tight', pad=0)




if __name__ == "__main__":
    main()