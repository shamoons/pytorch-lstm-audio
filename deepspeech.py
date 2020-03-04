from audio_util import load_audio_spectrogram
from submodules import DeepSpeech
import torch


def main():
    model = DeepSpeech.load_model(
        'data/pretrained_models/librispeech_pretrained_v2-2.pth')

    model.eval()
    model = model.to('cpu')
    # if use_half:
    # model = model.half()
    # return model

    # model = utils.load_model(
    #     'cpu', 'data/pretrained_models/librispeech_pretrained_v2-2.pth', False)
    print(model)
    print(model.audio_conf)

    spect = load_audio_spectrogram(
        'data/dev-noise-subtractive-250ms-1/1272/135031/1272-135031-0023.flac')

    print('spect', spect.shape)

    spect = spect.view(1, 1, spect.size(0), spect.size(1))
    input_sizes = torch.IntTensor([spect.size(3)]).int()
    print('spect.view', spect.shape)
    print('input_sizes', input_sizes, input_sizes.shape)
    out, output_sizes = model(spect, input_sizes)

    print(out, output_sizes)


if __name__ == '__main__':
    main()
