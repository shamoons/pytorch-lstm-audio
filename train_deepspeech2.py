import os

from submodules.deepspeech2.utils import reduce_tensor, check_loss
from submodules.deepspeech2.test import evaluate
from submodules.deepspeech2.model import DeepSpeech, supported_rnns
from submodules.deepspeech2.logger import VisdomLogger, TensorBoardLogger
from submodules.deepspeech2.decoder import GreedyDecoder
from submodules.deepspeech2.data.data_loader import AudioDataLoader, SpectrogramDataset, BucketingSampler
from torch.nn import CTCLoss
import torch.utils.data.distributed
import torch.distributed as dist
import numpy as np
import argparse
import json

import random
import time
import wandb
import socket

from utils.model_loader import load_masking_model
from reconstruction_model import ReconstructionModel


# from warpctc_pytorch import CTCLoss


parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--train-manifest', metavar='DIR',
                    help='path to train manifest csv', default='data/train_manifest.csv')
parser.add_argument('--val-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/val_manifest.csv')
parser.add_argument('--sample-rate', default=16000,
                    type=int, help='Sample rate')
parser.add_argument('--batch-size', default=20, type=int,
                    help='Batch size for training')
parser.add_argument('--num-workers', default=4, type=int,
                    help='Number of workers used in data-loading')
parser.add_argument('--labels-path', default='labels.json',
                    help='Contains all characters for transcription')
parser.add_argument('--window-size', default=.02, type=float,
                    help='Window size for spectrogram in seconds')
parser.add_argument('--window-stride', default=.01, type=float,
                    help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming',
                    help='Window type for spectrogram generation')
parser.add_argument('--hidden-size', default=800,
                    type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden-layers', default=5,
                    type=int, help='Number of RNN layers')
parser.add_argument('--rnn-type', default='gru',
                    help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--epochs', default=70, type=int,
                    help='Number of training epochs')
parser.add_argument('--cuda', dest='cuda', action='store_true',
                    help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=3e-4,
                    type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--max-norm', default=400, type=int,
                    help='Norm cutoff to prevent explosion of gradients')
parser.add_argument('--learning-anneal', default=1.1, type=float,
                    help='Annealing applied to learning rate every epoch')
parser.add_argument('--silent', dest='silent', action='store_true',
                    help='Turn off progress tracking per iteration')
parser.add_argument('--checkpoint', dest='checkpoint',
                    action='store_true', help='Enables checkpoint saving of model')
parser.add_argument('--checkpoint-per-batch', default=0, type=int,
                    help='Save checkpoint per batch. 0 means never save')
parser.add_argument('--visdom', dest='visdom',
                    action='store_true', help='Turn on visdom graphing')
parser.add_argument('--tensorboard', dest='tensorboard',
                    action='store_true', help='Turn on tensorboard graphing')
parser.add_argument('--log-dir', default='visualize/deepspeech_final',
                    help='Location of tensorboard log')
parser.add_argument('--log-params', dest='log_params',
                    action='store_true', help='Log parameter values and gradients')
parser.add_argument('--id', default='Deepspeech training',
                    help='Identifier for visdom/tensorboard run')
parser.add_argument('--save-folder', default='models/',
                    help='Location to save epoch models')
parser.add_argument('--model-path', default='models/deepspeech_final.pth',
                    help='Location to save best validation model')
parser.add_argument('--continue-from', default='',
                    help='Continue from checkpoint model')
parser.add_argument('--initialize-baseline', default='',
                    help='Initialize baseline model from')
parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='Finetune the model from checkpoint "continue_from"')
parser.add_argument('--speed-volume-perturb', dest='speed_volume_perturb',
                    action='store_true', help='Use random tempo and gain perturbations.')
parser.add_argument('--spec-augment', dest='spec_augment', action='store_true',
                    help='Use simple spectral augmentation on mel spectograms.')
parser.add_argument('--noise-dir', default=None,
                    help='Directory to inject noise into audio. If default, noise Inject not added')
parser.add_argument('--noise-prob', default=0.4,
                    help='Probability of noise being added per sample')
parser.add_argument('--noise-min', default=0.0,
                    help='Minimum noise level to sample from. (1.0 means all noise, not original signal)', type=float)
parser.add_argument('--noise-max', default=0.5,
                    help='Maximum noise levels to sample from. Maximum 1.0', type=float)
parser.add_argument('--no-shuffle', dest='no_shuffle', action='store_true',
                    help='Turn off shuffling and sample from dataset based on sequence length (smallest to largest)')
parser.add_argument('--no-sortaGrad', dest='no_sorta_grad', action='store_true',
                    help='Turn off ordering of dataset on sequence length for the first epoch.')
parser.add_argument('--no-bidirectional', dest='bidirectional', action='store_false', default=True,
                    help='Turn off bi-directional RNNs, introduces lookahead convolution')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--rank', default=0, type=int,
                    help='The rank of this process')
parser.add_argument('--gpu-rank', default=None,
                    help='If using distributed parallel for multi-gpu, sets the GPU for the process')
parser.add_argument('--seed', default=123456, type=int,
                    help='Seed to generators')
parser.add_argument('--loss-scale', type=str, default=1)
parser.add_argument(
    '--mask_wandb', help='Path for the trained masking model', required=True)


torch.manual_seed(123456)
torch.cuda.manual_seed_all(123456)


def loss_fn(log_probs, targets, input_lengths, target_lengths):
    criterion = torch.nn.CTCLoss(reduction='sum', zero_infinity=True)

    return criterion(log_probs, targets, input_lengths, target_lengths)


def decode_results(decoded_output, decoded_offsets):
    decoder = 'greedy'
    results = {
        "output": [],
        "_meta": {
            "acoustic_model": {
                "name": os.path.basename('data/pretrained_models/librispeech_pretrained_v2-2.pth')
            },
            "language_model": {
                "name": os.path.basename('data/saved_models/3-gram.pruned.3e-7.arpa'),
            },
            "decoder": {
                "lm": 'data/saved_models/3-gram.pruned.3e-7.arpa',
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


def to_np(x):
    return x.cpu().numpy()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main():
    args = parser.parse_args()

    wandb_tags = [socket.gethostname()]
    wandb.init(project="jstsp-reconstruction-with-deepspeech2",
               tags=','.join(wandb_tags))
    wandb.save('*.pt')

    # Set seeds for determinism
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    args.distributed = args.world_size > 1
    main_proc = True
    device = torch.device("cuda" if args.cuda else "cpu")

    save_folder = args.save_folder
    os.makedirs(save_folder, exist_ok=True)  # Ensure save folder exists

    loss_results, cer_results, wer_results = torch.Tensor(args.epochs), torch.Tensor(args.epochs), torch.Tensor(
        args.epochs)
    best_wer = None
    if main_proc and args.visdom:
        visdom_logger = VisdomLogger(args.id, args.epochs)
    if main_proc and args.tensorboard:
        tensorboard_logger = TensorBoardLogger(
            args.id, args.log_dir, args.log_params)

    avg_loss, start_epoch, start_iter, optim_state, amp_state = 0, 0, 0, None, None
    if args.continue_from:  # Starting from previous model
        print("Loading checkpoint model %s" % args.continue_from)
        package = torch.load(args.continue_from,
                             map_location=lambda storage, loc: storage)
        model = DeepSpeech.load_model_package(package)

        for param in model.parameters():
            param.requires_grad_(False)

        labels = model.labels
        audio_conf = model.audio_conf
        if not args.finetune:  # Don't want to restart training
            optim_state = package['optim_dict']
            # amp_state = package['amp']
            start_epoch = int(package.get('epoch', 1)) - \
                1  # Index start at 0 for training
            start_iter = package.get('iteration', None)
            if start_iter is None:
                # We saved model after epoch finished, start at the next epoch.
                start_epoch += 1
                start_iter = 0
            else:
                start_iter += 1
            avg_loss = int(package.get('avg_loss', 0))
            loss_results, cer_results, wer_results = package['loss_results'], package['cer_results'], \
                package['wer_results']
            best_wer = wer_results[start_epoch]
            if main_proc and args.visdom:  # Add previous scores to visdom graph
                visdom_logger.load_previous_values(start_epoch, package)
            if main_proc and args.tensorboard:  # Previous scores to tensorboard logs
                tensorboard_logger.load_previous_values(start_epoch, package)
    else:
        with open(args.labels_path) as label_file:
            labels = str(''.join(json.load(label_file)))

        audio_conf = dict(sample_rate=args.sample_rate,
                          window_size=args.window_size,
                          window_stride=args.window_stride,
                          window=args.window,
                          noise_dir=args.noise_dir,
                          noise_prob=args.noise_prob,
                          noise_levels=(args.noise_min, args.noise_max))

        rnn_type = args.rnn_type.lower()
        assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"
        model = DeepSpeech(rnn_hidden_size=args.hidden_size,
                           nb_layers=args.hidden_layers,
                           labels=labels,
                           rnn_type=supported_rnns[rnn_type],
                           audio_conf=audio_conf,
                           bidirectional=args.bidirectional)

    decoder = GreedyDecoder(labels)

    train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.train_manifest, labels=labels,
                                       normalize=False, speed_volume_perturb=args.speed_volume_perturb, spec_augment=args.spec_augment)
    test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.val_manifest, labels=labels,
                                      normalize=False, speed_volume_perturb=False, spec_augment=False)

    print('train_dataset', len(train_dataset))

    train_sampler = BucketingSampler(
        train_dataset, batch_size=args.batch_size)
    print('train_sampler', len(train_sampler), args.batch_size)
    train_loader = AudioDataLoader(train_dataset,
                                   num_workers=args.num_workers, batch_sampler=train_sampler)
    test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)

    if (not args.no_shuffle and start_epoch != 0) or args.no_sorta_grad:
        print("Shuffling batches for the following epochs")
        train_sampler.shuffle(start_epoch)

    reconstruct_model = ReconstructionModel(feature_dim=161, make_4d=True)
    mask_model = load_masking_model(args.mask_wandb, device, make_4d=True)

    # if args.initialize_baseline:
    #     print('Initializing Baseline:', args.initialize_baseline)
    #     baseline_state_dict = torch.load(args.initialize_baseline, map_location=device)
    #     baseline_m.load_state_dict(baseline_state_dict)

    # model = torch.nn.Sequential(baseline_m, model)

    reconstruct_model = reconstruct_model.to(device)
    mask_model = mask_model.to(device)
    model = model.to(device)
    parameters = model.parameters()
    optimizer = torch.optim.SGD(parameters, lr=args.lr,
                                momentum=args.momentum, nesterov=True, weight_decay=1e-5)
    baseline_optimizer = torch.optim.SGD(reconstruct_model.parameters(), lr=args.lr,
                                         momentum=args.momentum, nesterov=True, weight_decay=1e-5)

    if optim_state is not None:
        optimizer.load_state_dict(optim_state)

    print("Number of parameters: %d" % DeepSpeech.get_param_size(model))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        end = time.time()
        start_epoch_time = time.time()
        for i, (data) in enumerate(train_loader, start=start_iter):
            if i == len(train_sampler):
                break
            inputs, targets, input_percentages, target_sizes = data

            if args.cuda:
                inputs = inputs.cuda()

            print('Input', inputs.size(), inputs.min(), inputs.mean(), inputs.max())
            mask = mask_model(inputs)
            mask = torch.round(mask)

            expanded_mask = mask_model.expand_mask(mask, seq_length=inputs.size(3))

            print('Mask', mask.size(), mask.min(), mask.mean(), mask.max())
            print('ExpandedMask', expanded_mask.size(), expanded_mask.min(), expanded_mask.mean(), expanded_mask.max())
            masked_inputs = inputs * expanded_mask.unsqueeze(2)

            print('masked_inputs', masked_inputs.size(), masked_inputs.min(), masked_inputs.mean(), masked_inputs.max())
            # quit()

            reconstruct_output = reconstruct_model(masked_inputs)
            print('reconstruct_output', reconstruct_output.size(), reconstruct_output.min(), reconstruct_output.mean(), reconstruct_output.max())

            reconstruct_output[mask == 0] = inputs[mask == 0]

            input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
            # measure data loading time
            data_time.update(time.time() - end)
            inputs = inputs.to(device)

            out, output_sizes = model(reconstruct_output, input_sizes)
            decoded_output, decoded_offsets = decoder.decode(out, output_sizes)

            print(json.dumps(decode_results(decoded_output, decoded_offsets)))
            out = out.transpose(0, 1)  # TxNxH
            out = out.log_softmax(-1)

            float_out = out.float()  # ensure float32 for loss
            loss = loss_fn(float_out, targets.to(device).long(),
                           output_sizes.to(device), target_sizes.to(device))

            loss_value = loss.item()

            # Check to ensure valid loss was calculated
            valid_loss, error = check_loss(loss, loss_value)
            if valid_loss:
                optimizer.zero_grad()
                # compute gradient

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_norm)
                optimizer.step()
            else:
                print(error)
                print('Skipping grad update')
                loss_value = 0

            loss_value /= inputs.size(0)
            avg_loss += loss_value
            losses.update(loss_value, inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.silent:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                          (epoch + 1), (i + 1), len(train_sampler), batch_time=batch_time, data_time=data_time, loss=losses))
            if args.checkpoint_per_batch > 0 and i > 0 and (i + 1) % args.checkpoint_per_batch == 0 and main_proc:
                file_path = '%s/deepspeech_checkpoint_epoch_%d_iter_%d.pth' % (
                    save_folder, epoch + 1, i + 1)
                print("Saving checkpoint model to %s" % file_path)
                torch.save(DeepSpeech.serialize(model, optimizer=optimizer, amp=None, epoch=epoch, iteration=i,
                                                loss_results=loss_results,
                                                wer_results=wer_results, cer_results=cer_results, avg_loss=avg_loss),
                           file_path)
            del loss, out, float_out
            torch.save(reconstruct_model.state_dict(),
                       os.path.join(wandb.run.dir, 'latest-model.pt'))

        avg_loss /= len(train_sampler)

        epoch_time = time.time() - start_epoch_time
        print('Training Summary Epoch: [{0}]\t'
              'Time taken (s): {epoch_time:.0f}\t'
              'Average Loss {loss:.3f}\t'.format(epoch + 1, epoch_time=epoch_time, loss=avg_loss))

        start_iter = 0  # Reset start iteration for next epoch
        with torch.no_grad():
            wer, cer, output_data = evaluate(test_loader=test_loader,
                                             device=device,
                                             model=model,
                                             decoder=decoder,
                                             target_decoder=decoder)
        loss_results[epoch] = avg_loss
        wer_results[epoch] = wer
        cer_results[epoch] = cer
        print('Validation Summary Epoch: [{0}]\t'
              'Average WER {wer:.3f}\t'
              'Average CER {cer:.3f}\t'.format(
                  epoch + 1, wer=wer, cer=cer))

        values = {
            'loss_results': loss_results,
            'cer_results': cer_results,
            'wer_results': wer_results
        }

        # wandb.log({
        #     'avg_loss': avg_loss,
        #     'loss_results': loss_results,
        #     'cer_results': cer_results,
        #     'wer_results': wer_results
        # })

        if args.visdom and main_proc:
            visdom_logger.update(epoch, values)
        if args.tensorboard and main_proc:
            tensorboard_logger.update(epoch, values, model.named_parameters())
            values = {
                'Avg Train Loss': avg_loss,
                'Avg WER': wer,
                'Avg CER': cer
            }

        if main_proc and args.checkpoint:
            file_path = '%s/deepspeech_%d.pth.tar' % (save_folder, epoch + 1)
            torch.save(DeepSpeech.serialize(model, optimizer=optimizer, amp=None, epoch=epoch, loss_results=loss_results,
                                            wer_results=wer_results, cer_results=cer_results),
                       file_path)
        # anneal lr
        for g in optimizer.param_groups:
            g['lr'] = g['lr'] / args.learning_anneal
        print(
            'DeepSpeech Learning rate annealed to: {lr:.6f}'.format(lr=g['lr']))

        for g in baseline_optimizer.param_groups:
            g['lr'] = g['lr'] / args.learning_anneal

        print(
            'Baseline Learning rate annealed to: {lr:.6f}'.format(lr=g['lr']))

        if main_proc and (best_wer is None or best_wer > wer):
            print("Found better validated model, saving to %s" %
                  args.model_path)
            torch.save(reconstruct_model.state_dict(), os.path.join(
                wandb.run.dir, 'best-model.pt'))

            torch.save(DeepSpeech.serialize(model, optimizer=optimizer, amp=None, epoch=epoch, loss_results=loss_results,
                                            wer_results=wer_results, cer_results=cer_results), args.model_path)
            best_wer = wer
            avg_loss = 0

        if not args.no_shuffle:
            print("Shuffling batches...")
            train_sampler.shuffle(epoch)


if __name__ == '__main__':
    main()
