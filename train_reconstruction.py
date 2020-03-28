from audio_dataset import AudioDataset
from barbar import Bar
from masking_model import MaskingModel
from reconstruction_model import ReconstructionModel

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

    parser.add_argument(
        '--audio_path', help='Path for corrupted audio', required=True)

    parser.add_argument(
        '--masking_model', help='Path for the trained masking model', required=True)

    parser.add_argument('--seq_length', help='Length of sequences of the spectrogram',
                        nargs='+', type=int, default=[16, 256])

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

    parser.add_argument('--final_kernel_size', default=11,
                        type=int, help='Final kernel size')

    parser.add_argument('--kernel_size', default=25,
                        type=int, help='Kernel size start')

    parser.add_argument('--kernel_size_step', default=-4,
                        type=int, help='Kernel size step')

    parser.add_argument('--verbose', default=False,
                        type=bool, help='Verbose mode')

    parser.add_argument('--seed', default=1,
                        type=int, help='Seed')


    args = parser.parse_args()

    return args


def initialize(args):
    torch.set_default_tensor_type('torch.FloatTensor')
    wandb_tags = [socket.gethostname()]
    wandb.init(project="jstsp2020-reconstruction",
               tags=','.join(wandb_tags), config=args)
    wandb.save('*.pt')
    wandb.save('*.onnx')
    np.random.seed(args.seed)

def cos_similiarity_loss(inp, target):
    loss = 1 - torch.nn.CosineSimilarity(dim=2)(inp + 1, target + 1)
    loss = loss.mean()
    return loss

def main():
    args = parse_args()
    initialize(args)

    reconstruction_model_file = open('reconstruction_model.py', 'r').read()
    open(path.join(wandb.run.dir, 'reconstruction_model.py'), 'w').write(
        reconstruction_model_file)
    open(path.join(wandb.run.dir, 'args.json'),
         'w').write(json.dumps(vars(args)))
    wandb.save('reconstruction_model.py')
    wandb.save('args.json')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = {'pin_memory': True} if device == 'cuda' else {}

    train_set = AudioDataset(
        args.audio_path, train_set=True, seq_length=args.seq_length, batch_size=args.batch_size, feature_dim=args.feature_dim, repeat_sample=args.repeat_sample, shuffle=True, normalize=False, mask=False)

    val_set = AudioDataset(
        args.audio_path, test_set=True, seq_length=args.seq_length, batch_size=args.batch_size, feature_dim=args.feature_dim, shuffle=True, normalize=False, mask=False)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=1, num_workers=args.num_workers, **params)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=1, num_workers=args.num_workers, **params)

    data_loaders = {'train': train_loader, 'val': val_loader}

    (head, tail) = path.split(args.masking_model)
    mask_args_path = path.join(
        head, tail.replace('best-model.pt', 'args.json'))
    masked_args = json.loads(open(mask_args_path, 'r').read())

    mask_model = MaskingModel(feature_dim=masked_args['feature_dim'], kernel_size=masked_args['kernel_size'],
                              kernel_size_step=masked_args['kernel_size_step'], final_kernel_size=masked_args['final_kernel_size'], device=device)
    mask_state_dict = torch.load(args.masking_model, map_location=device)
    mask_model.load_state_dict(mask_state_dict)
    mask_model.eval()

    reconstruct_model = ReconstructionModel(feature_dim=args.feature_dim,
                                            verbose=args.verbose, kernel_size=args.kernel_size, kernel_size_step=args.kernel_size_step, final_kernel_size=args.final_kernel_size)

    if args.continue_from:
        state_dict = torch.load(args.continue_from, map_location=device)
        reconstruct_model.load_state_dict(state_dict)
        print('Loading saved model to continue from: {}'.format(args.continue_from))

    optimizer = optim.Adam(reconstruct_model.parameters(), lr=args.base_lr, weight_decay=0)

    loss_fn = torch.nn.MSELoss(reduction='mean')

    wandb.watch(reconstruct_model)

    current_best_validation_loss = 1
    reconstruct_model = reconstruct_model.float()

    if torch.cuda.is_available():
        reconstruct_model.cuda()
        mask_model.cuda()

    early_stop_count = 0
    last_val_loss = current_best_validation_loss

    saved_onnx = False

    for epoch in range(args.epochs):
        reconstruct_model.train(True)  # Set model to training mode

        if (epoch + 1) % (args.epochs // args.lr_bump_partition) == 0:
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * args.lr_bump
            print('Bumping up LR to: {lr:.3e}'.format(lr=g['lr']))

        start_time = time.time()
        train_running_loss = 0.0
        train_count = 0

        for _, data in enumerate(Bar(data_loaders['train'])):
            inputs = data[0][0]
            outputs = data[1][0]

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                outputs = outputs.cuda()

            optimizer.zero_grad()

            mask = mask_model(inputs)

            mask = torch.round(mask)

            if torch.sum(mask) == 0:
                print('continue')
                continue

            expanded_mask = mask_model.expand_mask(mask, seq_length=inputs.size(1))

            masked_inputs = inputs * expanded_mask[..., None]
            masked_outputs = outputs * expanded_mask[..., None]

            pred = reconstruct_model(masked_inputs)

            loss = loss_fn(pred, masked_outputs)
            # loss = cos_similiarity_loss(pred, masked_outputs)

            loss.backward()
            optimizer.step()

            train_running_loss += loss.data
            train_count += 1

            if not saved_onnx:
                torch.onnx.export(reconstruct_model, masked_inputs, path.join(
                    wandb.run.dir, 'best-model.onnx'), verbose=False)
                saved_onnx = True

            # print('\ninput\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}'.format(
            #     torch.mean(inputs), torch.std(inputs), torch.min(inputs), torch.max(inputs)))

            # print('\noutputs\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}'.format(
            #     torch.mean(outputs), torch.std(outputs), torch.min(outputs), torch.max(outputs)))

            # print('\npred\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}'.format(
            #     torch.mean(pred), torch.std(pred), torch.min(pred), torch.max(pred)))

        reconstruct_model.eval()
        val_running_loss = 0.0
        val_count = 0

    
        # for _, data in enumerate(data_loaders['val']):
        #     inputs = data[0][0]
        #     outputs = data[1][0]

        #     if torch.cuda.is_available():
        #         inputs = inputs.cuda()
        #         outputs = outputs.cuda()

        #     mask = mask_model(inputs)

        #     mask = torch.round(mask)
        #     if torch.sum(mask) == 0:
        #         continue

        #     expanded_mask = mask_model.expand_mask(mask, seq_length=inputs.size(1))
        #     masked_inputs = expanded_mask.unsqueeze(2) * inputs
        #     masked_outputs = expanded_mask.unsqueeze(2) * outputs

        #     pred = reconstruct_model(masked_inputs)

        #     loss = loss_fn(pred, masked_outputs)
        #     # loss = cos_similiarity_loss(pred, masked_outputs)

        #     val_running_loss += loss.data
        #     val_count += 1


        time_per_epoch = int(time.time() - start_time)
        train_loss = train_running_loss / train_count
        val_loss = train_loss
        # val_loss = val_running_loss / val_count

        wandb.log({
            "train_loss": train_loss,
            'val_loss': val_loss,
            'epoch': epoch,
            'sec_per_epoch': time_per_epoch,
        })

        print(f"Epoch: {epoch}\tLoss(t): {train_loss:.6g}\tLoss(v): {val_loss:.6g} (best: {current_best_validation_loss:.6g})\tTime (epoch): {time_per_epoch:d}s")

        if val_loss < current_best_validation_loss:
            torch.save(reconstruct_model.state_dict(), path.join(
                wandb.run.dir, 'best-model.pt'))
            current_best_validation_loss = val_loss
        torch.save(reconstruct_model.state_dict(), path.join(
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

        if early_stop_count == 50:
            print('Early stopping because no val_loss improvement for 50 epochs')
            break


if __name__ == '__main__':
    main()
