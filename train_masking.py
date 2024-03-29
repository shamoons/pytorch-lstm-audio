from audio_dataset import AudioDataset
from audio_dataset import pad_samples
from barbar import Bar
from masking_model import MaskingModel
from ignite.metrics import Accuracy, Precision
from utils.model import init_weights

import argparse
import os.path as path
import socket
import torch
import torch.optim as optim
import wandb
import json
import time
import re
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--audio_paths', help='Path for corrupted audio', required=True)

    parser.add_argument('--feature_dim', help='Feature dimension',
                        type=int, default=161)

    parser.add_argument('--base_lr',
                        help='Base learning rate', type=float, default=1e-4)

    parser.add_argument('--learning-anneal',
                        default=1.1, type=float,
                        help='Annealing applied to learning rate every epoch')

    parser.add_argument('--lr_bump', default=2, type=float,
                        help='Amount to bump up the learning rate by every lr_bump_patience epochs of no improvement')

    parser.add_argument('--lr_bump_patience', default=10, type=int,
                        help='Number of partitions for bumps')

    parser.add_argument('--epochs', help='Epochs to run',
                        type=int, default=250)

    parser.add_argument('--batch_size', help='Batch size',
                        type=int, default=64)

    parser.add_argument('--num_workers', help='Number of workers for data_loaders',
                        type=int, default=4)

    parser.add_argument('--continue-from', default='',
                        help='Continue from checkpoint model')

    parser.add_argument('--final_kernel_size',
                        default=25, type=int, help='Final kernel size')

    parser.add_argument('--kernel_size',
                        default=25, type=int, help='Kernel size start')

    parser.add_argument('--kernel_size_step',
                        default=-4, type=int, help='Kernel size step')

    parser.add_argument('--verbose', default=False,
                        type=bool, help='Verbose mode')

    args = parser.parse_args()

    return args


def initialize(args):
    torch.set_default_tensor_type('torch.FloatTensor')
    wandb_tags = [socket.gethostname()]
    wandb.init(project="jstsp2020-masking",
               tags=','.join(wandb_tags), config=args)
    wandb.save('*.pt')
    wandb.save('*.onnx')
    np.random.seed(0)


def hamming_distance(inp, target):
    # For the binary case, hamming is equivalent to L1
    fn = torch.nn.L1Loss(reduction='mean')
    return fn(inp, target)


def loss_fn(inp, target):
    fn = torch.nn.BCELoss(reduction='mean')
    raw_loss = fn(inp, target)
    rounded_loss = fn(torch.round(inp), target)

    loss = raw_loss + rounded_loss

    return loss


def main():
    args = parse_args()
    initialize(args)

    baseline_model_file = open('masking_model.py', 'r').read()
    open(path.join(wandb.run.dir, 'saved_masking_model.py'), 'w').write(
        baseline_model_file)
    open(path.join(wandb.run.dir, 'args.json'),
         'w').write(json.dumps(vars(args)))
    wandb.save('saved_masking_model.py')
    wandb.save('args.json')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = {'pin_memory': True} if device == 'cuda' else {}

    audio_paths = args.audio_paths.strip().split(',')
    train_set = AudioDataset(audio_paths, train_set=True, normalize=False, mask=True)

    val_set = AudioDataset(audio_paths, test_set=True, normalize=False, mask=True)

    train_loader = torch.utils.data.DataLoader(
        train_set, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=pad_samples, **params)
    val_loader = torch.utils.data.DataLoader(
        val_set, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=pad_samples, **params)

    data_loaders = {'train': train_loader, 'val': val_loader}

    model = MaskingModel(verbose=args.verbose, kernel_size=args.kernel_size, kernel_size_step=args.kernel_size_step, final_kernel_size=args.final_kernel_size, device=device)

    if args.continue_from:
        state_dict = torch.load(args.continue_from, map_location=device)
        model.load_state_dict(state_dict)
        print('Loading saved model to continue from: {}'.format(args.continue_from))

    optimizer = optim.Adam(
        model.parameters(), lr=args.base_lr, weight_decay=0)

    wandb.watch(model)

    current_best_validation_loss = 10
    model = model.float()

    if torch.cuda.is_available():
        model.cuda()

    early_stop_count = 0
    last_val_loss = current_best_validation_loss

    saved_onnx = False
    torch.set_printoptions(profile='full', precision=3,
                           sci_mode=False, linewidth=180)

    print(f"Training Samples: {len(train_set)}")
    print(f"Validation Samples: {len(val_set)}")

    for epoch in range(args.epochs):
        model.train(True)  # Set model to training mode

        start_time = time.time()
        train_running_loss = 0.0

        for _, data in enumerate(Bar(data_loaders['train'])):
            inputs = data[0]
            outputs = data[1]

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                outputs = outputs.cuda()

            optimizer.zero_grad()

            pred = model(inputs)

            loss = loss_fn(pred, outputs)

            loss.backward()
            optimizer.step()

            train_running_loss += loss.data

            if not saved_onnx:
                torch.onnx.export(model, inputs, path.join(
                    wandb.run.dir, 'best-model.onnx'), verbose=False)
                saved_onnx = True

            # print('\ninput\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}'.format(
            #     torch.mean(inputs), torch.std(inputs), torch.min(inputs), torch.max(inputs)))

            # print('\noutputs\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}'.format(
            #     torch.mean(outputs), torch.std(outputs), torch.min(outputs), torch.max(outputs)))

            # print('\npred\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}'.format(
            #     torch.mean(pred), torch.std(pred), torch.min(pred), torch.max(pred)))

        # print(pred[0])
        # print(outputs[0])

        model.eval()
        val_running_loss = 0.0
        rounded_val_running_loss = 0.0
        hamming_distance_running_loss = 0.0
        for _, data in enumerate(data_loaders['val']):
            inputs = data[0]
            outputs = data[1]

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                outputs = outputs.cuda()

            pred = model(inputs)

            loss = loss_fn(pred, outputs)

            val_running_loss += loss.data
            rounded_val_running_loss += loss_fn(
                torch.round(pred), outputs).data
            hamming_distance_running_loss = hamming_distance(
                torch.round(pred), outputs).data

            # pred_rounded = torch.tensor(pred)
            # pred_rounded[pred_rounded < 0.5] = 0
            # pred_rounded[pred_rounded >= 0.5] = 1
            # rounded_loss = cos_mse_similiarity_loss(pred_rounded, outputs)

            # print(f"Epoch: {epoch}\tLoss: {loss:.4g}\tRounded Loss: {rounded_loss:.4g}\tValRunningLoss: {val_running_loss:.4g}\tSize: {pred.size(1)}\tLen: {len(data_loaders['val'])}")

        time_per_epoch = int(time.time() - start_time)
        train_loss = train_running_loss / len(data_loaders['train'])
        val_loss = val_running_loss / len(data_loaders['val'])
        rounded_val_loss = rounded_val_running_loss / len(data_loaders['val'])
        hamming_distance_loss = hamming_distance_running_loss / len(data_loaders['val'])

        wandb.log({
            "train_loss": train_loss,
            'val_loss': val_loss,
            'epoch': epoch,
            'sec_per_epoch': time_per_epoch,
            'rounded_val_loss': rounded_val_loss,
            'hamming_distance': hamming_distance_loss
        })

        print(f"Epoch: {epoch}\tLoss(t): {train_loss:.6g}\tLoss(v): {val_loss:.6g} (best: {current_best_validation_loss:.6g})\tRounded Loss (v): {rounded_val_loss:.6g}\tHamming Distance (v): {hamming_distance_loss:.6g}\t Time (epoch): {time_per_epoch:d}s")

        if val_loss < current_best_validation_loss:
            torch.save(model.state_dict(), path.join(
                wandb.run.dir, 'best-model.pt'))
            current_best_validation_loss = val_loss
        torch.save(model.state_dict(), path.join(
            wandb.run.dir, 'latest-model.pt'))

        if val_loss < last_val_loss:
            early_stop_count = 0
        else:
            early_stop_count = early_stop_count + 1
        last_val_loss = val_loss

        if epoch % 5 == 0:
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] / args.learning_anneal
            print(
                'DeepSpeech Learning rate annealed to: {lr:.3e}'.format(lr=g['lr']))

        if early_stop_count == args.lr_bump_patience:
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * args.lr_bump
            print('Bumping up LR to: {lr:.3e}'.format(lr=g['lr']))
            continue

        if early_stop_count == 50:
            print('Early stopping because no val_loss improvement for 50 epochs')
            break


if __name__ == '__main__':
    main()
