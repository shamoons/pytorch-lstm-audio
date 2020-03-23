from audio_dataset import AudioDataset
from barbar import Bar
from baseline_model import BaselineModel

import argparse
import os.path as path
import socket
import torch
import torch.optim as optim
import wandb
import json
import time
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--audio_path', help='Path for corrupted audio',
                        default='data/dev-noise-subtractive-250ms-1')

    parser.add_argument('--seq_length', help='Length of sequences of the spectrogram',
                        nargs='+', type=int, default=[32])

    parser.add_argument('--feature_dim', help='Feature dimension',
                        type=int, default=161)

    parser.add_argument('--base_lr',
                        help='Base learning rate', type=float, default=1e-3)

    parser.add_argument('--learning-anneal',
                        default=1.1, type=float,
                        help='Annealing applied to learning rate every epoch')

    parser.add_argument('--lr_bump', default=5, type=float,
                        help='Amount to bump up the learning rate by every lr_bump_partition epochs')

    parser.add_argument('--lr_bump_partition', default=10, type=int,
                        help='Number of partitions for bumps')

    parser.add_argument('--epochs', help='Epochs to run',
                        type=int, default=250)

    parser.add_argument('--batch_size', help='Batch size',
                        type=int, default=64)

    parser.add_argument('--repeat_sample', help='How many times to sample each file',
                        type=int, default=1)

    parser.add_argument('--num_workers', help='Number of workers for data_loaders',
                        type=int, default=16)

    parser.add_argument('--continue-from', default='',
                        help='Continue from checkpoint model')

    parser.add_argument('--final_kernel_size',
                        default=11, type=int, help='Final kernel size')

    parser.add_argument('--verbose', default=False,
                        type=bool, help='Verbose mode')

    args = parser.parse_args()

    return args


def initialize(args):
    torch.set_default_tensor_type('torch.FloatTensor')
    wandb_tags = [socket.gethostname()]
    wandb.init(project="speech-reconstruction-baseline",
               tags=','.join(wandb_tags), config=args)
    wandb.save('*.pt')
    wandb.save('*.onnx')
    np.random.seed(0)


def main():
    args = parse_args()
    initialize(args)

    baseline_model_file = open('baseline_model.py', 'r').read()
    open(path.join(wandb.run.dir, 'saved_model.py'), 'w').write(
        baseline_model_file)
    open(path.join(wandb.run.dir, 'args.json'),
         'w').write(json.dumps(vars(args)))
    wandb.save('saved_model.py')
    wandb.save('args.json')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = {'pin_memory': True} if device == 'cuda' else {}

    train_set = AudioDataset(
        args.audio_path, train_set=True, seq_length=args.seq_length, batch_size=args.batch_size, feature_dim=args.feature_dim, repeat_sample=args.repeat_sample, shuffle=True, normalize=False)

    val_set = AudioDataset(
        args.audio_path, test_set=True, seq_length=args.seq_length, batch_size=args.batch_size, feature_dim=args.feature_dim, shuffle=True, normalize=False)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=1, num_workers=args.num_workers, **params)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=1, num_workers=args.num_workers, **params)

    data_loaders = {'train': train_loader, 'val': val_loader}

    model = BaselineModel(feature_dim=args.feature_dim,
                verbose=args.verbose, final_kernel_size=args.final_kernel_size)

    if args.continue_from:
        state_dict = torch.load(args.continue_from, map_location=device)
        model.load_state_dict(state_dict)
        print('Loading saved model to continue from: {}'.format(args.continue_from))

    optimizer = optim.Adam(
        model.parameters(), lr=args.base_lr, weight_decay=0)

    loss_fn = torch.nn.L1Loss(reduction='mean')

    wandb.watch(model)

    current_best_validation_loss = 1
    model = model.float()

    if(torch.cuda.is_available()):
        model.cuda()

    early_stop_count = 0
    last_val_loss = current_best_validation_loss
    for epoch in range(args.epochs):
        model.train(True)  # Set model to training mode

        if (epoch + 1) % (args.epochs // args.lr_bump_partition) == 0:
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * args.lr_bump
            print('Bumping up LR to: {lr:.3e}'.format(lr=g['lr']))

        start_time = time.time()
        train_running_loss = 0.0

        for _, data in enumerate(Bar(data_loaders['train'])):
            inputs = data[0]

            outputs = data[1]
            # outputs = inputs

            if(torch.cuda.is_available()):
                inputs = inputs.cuda()
                outputs = outputs.cuda()

            optimizer.zero_grad()

            pred = model(inputs)

            loss = loss_fn(pred, outputs)

            loss.backward()
            optimizer.step()

            train_running_loss += loss.data
        # print('\ninput\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}'.format(
        #     torch.mean(inputs), torch.std(inputs), torch.min(inputs), torch.max(inputs)))

        # print('\npred\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}'.format(
        #     torch.mean(pred), torch.std(pred), torch.min(pred), torch.max(pred)))

        model.eval()
        val_running_loss = 0.0
        for _, data in enumerate(data_loaders['val']):
            inputs = data[0]
            outputs = data[1]
            # outputs = inputs

            if(torch.cuda.is_available()):
                inputs = inputs.cuda()
                outputs = outputs.cuda()

            pred = model(inputs)

            loss = loss_fn(pred, outputs)
            val_running_loss += loss.data

        time_per_epoch = int(time.time() - start_time)
        train_loss = train_running_loss / len(data_loaders['train'])
        val_loss = val_running_loss / len(data_loaders['val'])

        wandb.log({
            "train_loss": train_loss,
            'val_loss': val_loss,
            'epoch': epoch,
            'sec_per_epoch': time_per_epoch,
        })
        print('\tEpoch: {}\tLoss: {:.4g}\tVal Loss: {:.4g}\tTime per Epoch: {:.4g}s\n'.format(
            epoch, train_loss, val_loss, time_per_epoch))

        if val_loss < current_best_validation_loss:
            print('Saving new best model with val loss: ',
                  val_loss, '\tOld Loss was: ', current_best_validation_loss)
            torch.save(model.state_dict(), path.join(
                wandb.run.dir, 'best-model.pt'))
            torch.onnx.export(model, inputs, path.join(
                wandb.run.dir, 'best-model.onnx'), verbose=False)
            current_best_validation_loss = val_loss
        torch.save(model.state_dict(), path.join(
            wandb.run.dir, 'latest-model.pt'))
        torch.onnx.export(model, inputs, path.join(
            wandb.run.dir, 'latest-model.onnx'), verbose=False)

        if val_loss <= last_val_loss:
            early_stop_count = 0
        else:
            early_stop_count = early_stop_count + 1
        last_val_loss = val_loss

        for g in optimizer.param_groups:
            g['lr'] = g['lr'] / args.learning_anneal
        print('DeepSpeech Learning rate annealed to: {lr:.3e}'.format(lr=g['lr']))

        if early_stop_count == 50:
            print('Early stopping because no val_loss improvement for 50 epochs')
            break


if __name__ == '__main__':
    main()
