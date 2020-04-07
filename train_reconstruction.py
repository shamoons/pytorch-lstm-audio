from audio_dataset import AudioDataset
from audio_dataset import pad_samples
from barbar import Bar
from reconstruction_model import ReconstructionModel
from utils.model_loader import load_masking_model

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
        '--mask_wandb', help='Path for the trained masking model', required=True)

    parser.add_argument('--feature_dim', help='Feature dimension',
                        type=int, default=161)

    parser.add_argument('--base_lr',
                        help='Base learning rate', type=float, default=1e-3)

    parser.add_argument('--learning-anneal',
                        default=1.1, type=float,
                        help='Annealing applied to learning rate every epoch')

    parser.add_argument('--lr_bump', default=2, type=float,
                        help='Amount to bump up the learning rate by every lr_bump_patience epochs')

    parser.add_argument('--lr_bump_patience', default=10, type=int,
                        help='Number of partitions for bumps')

    parser.add_argument('--epochs', help='Epochs to run',
                        type=int, default=250)

    parser.add_argument('--batch_size', help='Batch size',
                        type=int, default=64)

    parser.add_argument('--repeat_sample', help='How many times to sample each file',
                        type=int, default=1)

    parser.add_argument('--num_workers', help='Number of workers for data_loaders',
                        type=int, default=4)

    parser.add_argument('--continue-from', default='',
                        help='Continue from checkpoint model')

    parser.add_argument('--final_kernel_size', default=25,
                        type=int, help='Final kernel size')

    parser.add_argument('--kernel_size', default=25,
                        type=int, help='Kernel size start')

    parser.add_argument('--kernel_size_step', default=-4,
                        type=int, help='Kernel size step')

    parser.add_argument('--verbose', default=False,
                        type=bool, help='Verbose mode')

    parser.add_argument('--seed', default=5,
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


def loss_fn(inp, target, mask, loss_weights):
    max_mask_len = torch.max(torch.sum(mask, 1)).int()

    # print(f"\n\ninp.size(): {inp.size()}")
    # print(f"target.size(): {target.size()}")
    # print(f"mask.size(): {mask.size()}")
    # print(f"max_mask_len: {max_mask_len}")

    non_zero_targets = []
    non_zero_masks = []
    for batch_index, mask_batch in enumerate(mask):
        mask_len = torch.sum(mask_batch).int()

        if mask_len == 0:
            continue
        non_zero_targets.append(target[batch_index])
        non_zero_masks.append(mask[batch_index])

    non_zero_targets = torch.stack(non_zero_targets)
    non_zero_masks = torch.stack(non_zero_masks)
    # print(f"non_zero_targets: {non_zero_targets.size()}")
    # print(f"non_zero_masks: {non_zero_masks.size()}")
    
    losses = []
    for batch_index, non_zero_mask in enumerate(non_zero_masks):
        m_nonzero = non_zero_mask.nonzero().flatten()
        first_nonzero = m_nonzero[0]
        last_nonzero = min(m_nonzero[-1], first_nonzero + max_mask_len - 1)    # TODO: See why this plus 1 is needed

        # print(f"first_nonzero: {first_nonzero}\tlast_nonzero: {last_nonzero}")
        # print(non_zero_targets[batch_index].size())
        batch_target = non_zero_targets[batch_index][first_nonzero:last_nonzero]

        # print(f"pre pad batch_taret: {batch_target.size()}")
        if batch_target.size(0) < inp[batch_index].size(0):
            pad_zeros = torch.zeros((inp[batch_index].size(0) - batch_target.size(0), batch_target.size(1))).to(mask.device)
            batch_target = torch.cat((batch_target, pad_zeros), 0)


        # print(f"inp: {inp[batch_index].size()}\t batch_target: {batch_target.size()}")
        loss = torch.nn.MSELoss(reduction='none')(inp[batch_index], batch_target)
        losses.append(loss)

    batch_loss = torch.stack(losses).sum() / torch.sum(mask)

    return batch_loss, 0
    # print(batch_loss)


    
    quit()
    masked_inp = mask.unsqueeze(2) * inp
    masked_target = mask.unsqueeze(2) * target

    loss = torch.nn.MSELoss(reduction='none')(masked_inp, masked_target)
    channel_loss = torch.sum(loss, dim=1)
    channel_loss = torch.mean(channel_loss, dim=0)

    # torch.set_printoptions(precision=2, sci_mode=True)
    # print(loss.size())
    # print('channel_loss', channel_loss.size())
    # print(channel_loss)
    # print('loss_weights', loss_weights.sum())
    # print(loss_weights)

    weighted_loss = channel_loss * (1 + loss_weights)

    softmax_weights = torch.nn.Softmax()(channel_loss).detach()

    # print('\n', channel_loss, '\n', weighted_loss, '\n',
    #       loss_weights, '\n', softmax_weights, '\n\n')

    # Instead of dividing by the number of elements. Divide by the number of non-zero mask elements in the batch
    # loss = loss.sum() / torch.sum(mask)
    loss = weighted_loss.sum() / torch.sum(mask)

    # print(softmax_weights.min(), softmax_weights.mean(), softmax_weights.max())
    # quit()
    return loss, softmax_weights


def main():
    args = parse_args()
    initialize(args)

    reconstruction_model_file = open('reconstruction_model.py', 'r').read()
    open(path.join(wandb.run.dir, 'saved_reconstruction_model.py'), 'w').write(
        reconstruction_model_file)
    open(path.join(wandb.run.dir, 'args.json'),
         'w').write(json.dumps(vars(args)))
    wandb.save('saved_reconstruction_model.py')
    wandb.save('args.json')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = {'pin_memory': True} if device == 'cuda' else {}

    train_set = AudioDataset(
        args.audio_path, train_set=True, feature_dim=args.feature_dim, repeat_sample=args.repeat_sample, normalize=False, mask=False)

    val_set = AudioDataset(
        args.audio_path, test_set=True, feature_dim=args.feature_dim, normalize=False, mask=False)

    train_loader = torch.utils.data.DataLoader(
        train_set, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=pad_samples, **params)
    val_loader = torch.utils.data.DataLoader(
        val_set, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=pad_samples, **params)

    data_loaders = {'train': train_loader, 'val': val_loader}

    mask_model = load_masking_model(args.mask_wandb, device)

    reconstruct_model = ReconstructionModel(feature_dim=args.feature_dim,
                                            verbose=args.verbose, kernel_size=args.kernel_size, kernel_size_step=args.kernel_size_step, final_kernel_size=args.final_kernel_size)

    # reconstruct_model.model_summary(reconstruct_model)
    if args.continue_from:
        state_dict = torch.load(args.continue_from, map_location=device)
        reconstruct_model.load_state_dict(state_dict)
        print('Loading saved model to continue from: {}'.format(args.continue_from))

    optimizer = optim.Adam(reconstruct_model.parameters(),
                           lr=args.base_lr, weight_decay=0)

    wandb.watch(reconstruct_model)

    current_best_validation_loss = 10
    reconstruct_model = reconstruct_model.float()

    if torch.cuda.is_available():
        reconstruct_model.cuda()
        mask_model.cuda()

    early_stop_count = 0
    last_val_loss = current_best_validation_loss

    saved_onnx = False

    torch.set_printoptions(profile='full', precision=3,
                           sci_mode=False, linewidth=180)
    print(f"Training Samples: {len(train_set)}")
    print(f"Validation Samples: {len(val_set)}")

    loss_weights = 0
    for epoch in range(args.epochs):
        reconstruct_model.train(True)  # Set model to training mode

        start_time = time.time()
        train_running_loss = 0.0
        train_count = 0
        for _, data in enumerate(Bar(data_loaders['train'])):
            inputs = data[0]
            outputs = data[1]

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                outputs = outputs.cuda()

            optimizer.zero_grad()

            mask = mask_model(inputs)
            mask = torch.round(mask)
            # expanded_mask = mask_model.expand_mask(
            #     mask, seq_length=inputs.size(1))

            # masked_inputs = inputs * expanded_mask[..., None]
            # masked_outputs = outputs * expanded_mask[..., None]

            pred = reconstruct_model(inputs, mask)
            # masked_outputs = reconstruct_model.get_nonzero_masked_outputs(mask, outputs)
            # print(f"mask: {mask.size()}")
            # print(f"inputs: {inputs.size()}")
            # print(f"pred: {pred.size()}")
            # print(f"masked_outputs: {masked_outputs.size()}")
            # quit()

            loss, loss_weights = loss_fn(
                pred, outputs, mask, loss_weights=loss_weights)

            loss.backward()
            optimizer.step()

            train_running_loss += loss.data
            train_count += 1

            # if not saved_onnx:
            #     torch.onnx.export(reconstruct_model, (inputs, mask), path.join(
            #         wandb.run.dir, 'best-model.onnx'), verbose=False)
            #     saved_onnx = True

        reconstruct_model.eval()
        val_running_loss = 0.0
        val_count = 0
        for _, data in enumerate(data_loaders['val']):
            inputs = data[0]
            outputs = data[1]

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                outputs = outputs.cuda()

            mask = mask_model(inputs)
            mask = torch.round(mask)

            # expanded_mask = mask_model.expand_mask(
            #     mask, seq_length=inputs.size(1))
            # masked_inputs = expanded_mask.unsqueeze(2) * inputs
            # masked_outputs = expanded_mask.unsqueeze(2) * outputs

            pred = reconstruct_model(inputs, mask)

            loss, _ = loss_fn(pred, outputs, mask, 0)

            val_running_loss += loss.data
            val_count += 1

        time_per_epoch = int(time.time() - start_time)
        train_loss = train_running_loss / train_count
        val_loss = val_running_loss / val_count

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

        if early_stop_count == args.lr_bump_patience:
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * args.lr_bump
            print('Bumping up LR to: {lr:.3e}'.format(lr=g['lr']))
            continue

        if epoch % 3 == 0:
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] / args.learning_anneal
            print(
                'DeepSpeech Learning rate annealed to: {lr:.3e}'.format(lr=g['lr']))

        if early_stop_count == 50:
            print('Early stopping because no val_loss improvement for 50 epochs')
            break


if __name__ == '__main__':
    main()
