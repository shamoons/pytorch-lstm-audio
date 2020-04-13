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
from utils.model_loader import load_masking_model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path", help="Model to Load", type=str)

    parser.add_argument('--mask_wandb', type=str,
                        help='Run mask based on a wandb id')

    parser.add_argument('--wandb', type=str, help='Run based on a wandb id')

    parser.add_argument("--audio_path", help="Audio file", type=str)

    parser.add_argument('--saved_model_path', help='File of the saved model',
                        default='saved_models')

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device("cpu") # TODO: Use CUDA when available
    mask_model = load_masking_model(args.mask_wandb, device)

    wandb_dir = list(glob.iglob(os.path.join(
        'wandb', '*' + args.wandb), recursive=False))[0]
    model_path = os.path.join(wandb_dir, 'best-model.pt')
    saved_model_path = wandb_dir
    args_path = os.path.join(wandb_dir, 'args.json')
    saved_args = json.loads(open(args_path, 'r').read())

    sys.path.append(os.path.abspath(saved_model_path))
    model = importlib.import_module('saved_reconstruction_model').ReconstructionModel(
        feature_dim=saved_args['feature_dim'], kernel_size=saved_args['kernel_size'], kernel_size_step=saved_args['kernel_size_step'])

    state_dict = torch.load(model_path, map_location=device)

    model.load_state_dict(state_dict)

    model = model.float()
    model.eval()

    filename_without_ext = Path(args.audio_path).stem

    # clean_audio_path = re.sub(f'{pattern}[^/]+/', f'{pattern}clean/', args.audio_path)

    input_spectrogram, samples_length, sample_rate, n_fft, hop_length = load_audio_spectrogram(
        args.audio_path)

    input_spectrogram = input_spectrogram.view(
        1, input_spectrogram.size(0), input_spectrogram.size(1))


    print('input_spectrogram\tMean: {:.4f} ± {:.4f}\tMin: {:.4f}\tMax: {:.4f}\tSize: {}'.format(torch.mean(input_spectrogram), torch.std(
        input_spectrogram), torch.min(input_spectrogram), torch.max(input_spectrogram), input_spectrogram.size()))


    if str(device) == 'cuda':
        mask_model.cuda()
        model.cuda()
        input_spectrogram = input_spectrogram.cuda()

    mask = mask_model(input_spectrogram)
    mask = torch.round(mask).float()
    mask_sum = torch.sum(mask).int()

    # Model takes data of shape: torch.Size([BATCH_SIZE, SEQUENCE_LENGTH, FEATURE_DIM])
    pred = model(input_spectrogram, mask)

    print(f"pred: {pred.size()}\tmask: {mask_sum}\tmask size: {mask.size()}")

    pred_t = pred.permute(0, 2, 1)

    pred = torch.nn.functional.interpolate(pred_t, size=mask_sum.item()).permute(0, 2, 1)

    output = input_spectrogram
    output[mask == 1] = pred
    
    torch.set_printoptions(profile='full', precision=3,
                           sci_mode=False, linewidth=180)

    # print(pred)

    # output[mask == 0] = input_spectrogram[mask == 0]

    print('pred\t\tMean: {:.4f} ± {:.4f}\tMin: {:.4f}\tMax: {:.4f}\tSize: {}'.format(
        torch.mean(pred), torch.std(pred), torch.min(pred), torch.max(pred), pred.size()))

    print('model output\t\tMean: {:.4f} ± {:.4f}\tMin: {:.4f}\tMax: {:.4f}\tSize: {}'.format(
        torch.mean(output), torch.std(output), torch.min(output), torch.max(output), output.size()))

    # diff = torch.abs(output - input_spectrogram)

    output = torch.expm1(output)
    # print('expm1 output\t\tMean: {:.4f} ± {:.4f}\tMin: {:.4f}\tMax: {:.4f}\tSize: {}'.format(
    #     torch.mean(output), torch.std(output), torch.min(output), torch.max(output), output.size()))

    # augmented_mask = torch.tensor(mask)
    # augmented_mask[augmented_mask ==
    #                1] = augmented_mask[augmented_mask == 1] + 0.1111

    # print('Mask')
    # print(augmented_mask[0])


    # print('Input')
    # print(torch.mean(input_spectrogram, 2)[0])

    # print('Output')
    # output = torch.Tensor(output)
    # print(torch.mean(output, 2)[0])

    # print('Diff')
    # print(torch.mean(diff, 2)[0])

    np_output = output.view(output.size(1), output.size(2)).detach().numpy()

    # output requires shape of [SEQUENCE_LEN, FEATURE_DIM]
    audio = create_audio_from_spectrogram(
        np_output, n_fft=n_fft, hop_length=hop_length, length=samples_length)

    sf.write(filename_without_ext + '.flac', audio, sample_rate)


if __name__ == '__main__':
    main()