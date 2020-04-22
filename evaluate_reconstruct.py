import json
import re
import importlib
import sys
import argparse
import torch
import os
import glob
from pathlib import Path
import soundfile as sf
from utils.audio_util import load_audio_spectrogram, create_audio_from_spectrogram
import numpy as np
# from utils.model_loader import load_masking_model, load_reconstruction_model
from deep_restore import DeepRestore


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mask_wandb', type=str, help='Run mask based on a wandb id')

    parser.add_argument('--wandb', type=str, help='Run based on a wandb id')

    parser.add_argument("--audio_path", help="Audio file", type=str)

    args = parser.parse_args()

    return args


def main():
    torch.set_printoptions(profile='full', precision=5,
                           sci_mode=False, linewidth=180)

    args = parse_args()
    filename_without_ext = Path(args.audio_path).stem

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device("cpu")  # TODO: Use CUDA when available

    deep_restore = DeepRestore(mask_wandb=args.mask_wandb, reconstruct_wandb=args.wandb, device=device)

    audio_signal, samplerate = sf.read(args.audio_path)
    print(f"audio_signal:\tmin: {audio_signal.min():.2f}\tmean: {audio_signal.mean():.2f} ± {audio_signal.std():.2f}\tmax: {audio_signal.max():.2f}")

    enhanced_signal = deep_restore.enhance(audio_signal)
    print(f"enhanced_signal:\tmin: {enhanced_signal.min():.2f}\tmean: {enhanced_signal.mean():.2f} ± {enhanced_signal.std():.2f}\tmax: {enhanced_signal.max():.2f}")
    sf.write(filename_without_ext + '.flac', enhanced_signal, samplerate)

    # print(enhanced_signal)
    quit()

    # mask_model = load_masking_model(args.mask_wandb, device)
    # model = load_reconstruction_model(args.wandb, device)

    # filename_without_ext = Path(args.audio_path).stem

    # pattern = 'dev-'
    # clean_audio_path = re.sub(f'{pattern}[^/]+/', f'{pattern}clean/', args.audio_path)
    # print(clean_audio_path)

    # input_spectrogram, samples_length, sample_rate, n_fft, hop_length = load_audio_spectrogram(
    #     args.audio_path)

    # input_spectrogram = input_spectrogram.view(
    #     1, input_spectrogram.size(0), input_spectrogram.size(1))

    # print('input_spectrogram\tMean: {:.6f} ± {:.6f}\tMin: {:.6f}\tMax: {:.6f}\tSize: {}'.format(torch.mean(input_spectrogram), torch.std(
    #     input_spectrogram), torch.min(input_spectrogram), torch.max(input_spectrogram), input_spectrogram.size()))

    # if str(device) == 'cuda':
    #     mask_model.cuda()
    #     model.cuda()
    #     input_spectrogram = input_spectrogram.cuda()

    # mask = mask_model(input_spectrogram)
    # mask = torch.round(mask).float()

    # # Model takes data of shape: torch.Size([BATCH_SIZE, SEQUENCE_LENGTH, FEATURE_DIM])
    # pred = model(input_spectrogram, mask)
    # pred = model.fit_to_size(pred, torch.sum(mask, 1).int().tolist())

    # output = input_spectrogram
    # print(f"output: {output.size()}")
    # print(f"pred: {pred.size()}")
    # print(f"mask: {mask.size()}")
    # output[mask == 1] = pred

    # print('pred\t\tMean: {:.6f} ± {:.6f}\tMin: {:.6f}\tMax: {:.6f}\tSize: {}'.format(
    #     torch.mean(pred), torch.std(pred), torch.min(pred), torch.max(pred), pred.size()))

    # print('spect output\t\tMean: {:.6f} ± {:.6f}\tMin: {:.6f}\tMax: {:.6f}\tSize: {}'.format(
    #     torch.mean(output), torch.std(output), torch.min(output), torch.max(output), output.size()))

    # # diff = torch.abs(output - input_spectrogram)

    # output = torch.expm1(output)
    # # print('expm1 output\t\tMean: {:.4f} ± {:.4f}\tMin: {:.4f}\tMax: {:.4f}\tSize: {}'.format(
    # #     torch.mean(output), torch.std(output), torch.min(output), torch.max(output), output.size()))

    # # print('Input')
    # # print(torch.mean(input_spectrogram, 2)[0])

    # # print('Output')
    # # output = torch.Tensor(output)
    # # print(torch.mean(output, 2)[0])

    # # print('Diff')
    # # print(torch.mean(diff, 2)[0])

    # np_output = output.view(output.size(1), output.size(2)).detach().numpy()

    # # output requires shape of [SEQUENCE_LEN, FEATURE_DIM]
    # audio = create_audio_from_spectrogram(
    #     np_output, n_fft=n_fft, hop_length=hop_length, length=samples_length)

    # sf.write(filename_without_ext + '.flac', audio, sample_rate)


if __name__ == '__main__':
    main()
