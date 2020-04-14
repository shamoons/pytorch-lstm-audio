import torch
import numpy as np
from model_loader import load_masking_model, load_reconstruction_model
from audio_util import convert_to_spectrogram, create_audio_from_spectrogram

class DeepRestore:
    def __init__(self, mask_wandb, reconstruct_wandb, device):
        torch.manual_seed(0)

        self.mask_model = load_masking_model(mask_wandb, device)
        self.reconstruct_model = load_reconstruction_model(reconstruct_wandb, device)
    
    def enhance(self, audio_signal):
        input_spectrogram, sample_rate, n_fft, hop_length = load_audio_spectrogram(audio_signal)

        input_spectrogram = input_spectrogram.view(1, input_spectrogram.size(0), input_spectrogram.size(1))


        mask = self.mask_model(input_spectrogram)
        mask = torch.round(mask).float()
        mask_sum = torch.sum(mask).int()

        pred = model(input_spectrogram, mask)

        pred_t = pred.permute(0, 2, 1)

        pred = torch.nn.functional.interpolate(pred_t, size=mask_sum.item()).permute(0, 2, 1)

        output = input_spectrogram
        output[mask == 1] = pred

        output = torch.expm1(output)

        np_output = output.view(output.size(1), output.size(2)).detach().numpy()

        enhanced_signal = create_audio_from_spectrogram(np_output, n_fft=n_fft, hop_length=hop_length, length=len(audio_signal))

        return enhanced_signal
