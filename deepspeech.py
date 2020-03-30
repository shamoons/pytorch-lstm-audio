import os
import json
from utils.audio_util import load_audio_spectrogram
from submodules import DeepSpeech
from submodules import GreedyDecoder
import torch
import numpy as np

model_path = 'data/pretrained_models/librispeech_pretrained_v2.pth'
lm_path = 'data/saved_models/3-gram.pruned.3e-7.arpa'


def decode_results(decoded_output, decoded_offsets):
    decoder = 'greedy'
    results = {
        "output": [],
        "_meta": {
            "acoustic_model": {
                "name": os.path.basename(model_path)
            },
            "language_model": {
                "name": os.path.basename(lm_path) if lm_path else None,
            },
            "decoder": {
                "lm": lm_path is not None,
                "alpha": 1.97,
                "beta": 4.36,
                "type": decoder,
            }
        }
    }

    for b in range(len(decoded_output)):
        for pi in range(min(1, len(decoded_output[b]))):
            # for pi in range(min(args.top_paths, len(decoded_output[b]))):
            result = {'transcription': decoded_output[b][pi]}
            # if args.offsets:
            #     result['offsets'] = decoded_offsets[b][pi].tolist()
            results['output'].append(result)
    return results


def transcribe():
    audio_path = 'data/dev-clean/1462/170142/1462-170142-0001.flac'
    model = DeepSpeech.load_model(model_path)
    decoder = GreedyDecoder(
        model.labels, blank_index=model.labels.index('_'))

    model.eval()
    model = model.to('cpu')
    # if use_half:
    # model = model.half()
    # return model

    # model = utils.load_model(
    #     'cpu', 'data/pretrained_models/librispeech_pretrained_v2.pth', False)
    print(model)
    print(model.audio_conf)
    quit()

    spect, _, _, _, _ = load_audio_spectrogram(
        audio_path, transpose=False, normalize_spect=True)

    spect = spect.view(1, 1, spect.size(0), spect.size(1))
    print(spect.size(), spect.mean(), spect.std())
    input_sizes = torch.IntTensor([spect.size(3)]).int()

    out, output_sizes = model(spect, input_sizes)

    decoded_output, decoded_offsets = decoder.decode(out, output_sizes)

    print(json.dumps(decode_results(decoded_output, decoded_offsets)))


if __name__ == '__main__':
    transcribe()
