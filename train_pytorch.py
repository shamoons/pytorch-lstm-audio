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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--audio_path', help='Path for corrupted audio',
                        default='data/dev-noise-subtractive-250ms-1')

    parser.add_argument(
        "--hidden_size", help="Hidden size for the LSTM Layers", type=int, default=20)

    parser.add_argument(
        "--num_layers", help="Number of layers in the model", type=int, default=2)

    parser.add_argument('--seq_length', help='Length of sequences of the spectrogram',
                        type=int, default=32)
    parser.add_argument('--feature_dim', help='Feature dimension',
                        type=int, default=161)

    parser.add_argument(
        '--base_lr', help='Base learning rate', type=float, default=4e-4)

    parser.add_argument(
        '--step_size_up', help='Amount of steps to upcycle the Cyclic Learning Rate', type=int, default=10)

    parser.add_argument(
        '--step_size_down', help='Amount of steps to downcycle the Cyclic Learning Rate', type=int, default=50)

    parser.add_argument('--epochs', help='Epochs to run',
                        type=int, default=250)

    parser.add_argument('--batch_size', help='Batch size',
                        type=int, default=64)

    parser.add_argument('--repeat_sample', help='How many times to sample each file',
                        type=int, default=1)

    parser.add_argument('--num_workers', help='Number of workers for data_loaders',
                        type=int, default=10)

    parser.add_argument('--continue-from', default='',help='Continue from checkpoint model')

    args = parser.parse_args()

    return args


def initialize():
    torch.set_default_tensor_type('torch.FloatTensor')
    wandb_tags = [socket.gethostname()]
    wandb.init(project="speech-reconstruction-baseline",
               tags=','.join(wandb_tags))
    wandb.save('*.pt')


def main():
    args = parse_args()
    initialize()

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
        args.audio_path, train_set=True, seq_length=args.seq_length, feature_dim=args.feature_dim, repeat_sample=args.repeat_sample, normalize=False)

    val_set = AudioDataset(
        args.audio_path, test_set=True, seq_length=args.seq_length, feature_dim=args.feature_dim, normalize=False)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, **params)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, **params)

    data_loaders = {'train': train_loader, 'val': val_loader}

    # model = BaselineModel(feature_dim=args.feature_dim,
    #                       hidden_size=args.hidden_size, seq_length=args.seq_length, num_layers=args.num_layers, dropout=0.5)
    model = BaselineModel(feature_dim=args.feature_dim, initialize_weights=False)
    if args.continue_from:
        state_dict = torch.load(args.continue_from, map_location=device)
        model.load_state_dict(state_dict)
        print('Loading saved model to continue from: {}'.format(args.continue_from))

    # optimizer = optim.SGD(model.parameters(), lr=args.base_lr,
    #                       momentum=0.9, weight_decay=0.1)

    optimizer = optim.Adam(
        model.parameters(), lr=args.base_lr, weight_decay=0.001)

    # scheduler = optim.lr_scheduler.CyclicLR(
    #     optimizer, base_lr=args.base_lr, max_lr=args.max_lr, mode='exp_range', step_size_up=args.step_size_up, step_size_down=args.step_size_down, gamma=args.gamma)

    loss_fn = torch.nn.MSELoss(reduction='mean')

    wandb.watch(model)

    current_best_validation_loss = 1
    model = model.float()

    if(torch.cuda.is_available()):
        model.cuda()

    early_stop_count = 0
    last_val_loss = current_best_validation_loss
    for epoch in range(args.epochs):
        model.train(True)  # Set model to training mode
        start_time = time.time()
        train_running_loss = 0.0
        for _, data in enumerate(Bar(data_loaders['train'])):
            # Keeping the hidden re-init here because each iteration is a batch of sequences
            # and batches are unrelated
            # hidden = model.init_hidden(args.batch_size)

            inputs = data[0]
            # print('inputs.size', inputs.size())
            outputs = data[1]
            outputs = inputs
            if(torch.cuda.is_available()):
                inputs = inputs.cuda()
                outputs = outputs.cuda()
                # if type(hidden) is tuple:
                #     hidden = tuple(map(lambda h: h.cuda(), hidden))
                # else:
                #     hidden = hidden.cuda()

            optimizer.zero_grad()

            # pred, hidden = model(inputs, hidden)
            # print('\ninput\tMean: {:.4f}\tSTD: {:.4f}\tMin: {:.4f}\tMax: {:.4f}'.format(torch.mean(inputs), torch.std(inputs), torch.min(inputs), torch.max(inputs)))
            pred = model(inputs)
            # print('\npred\tMean: {:.4f}\tSTD: {:.4f}\tMin: {:.4f}\tMax: {:.4f}'.format(torch.mean(pred), torch.std(pred), torch.min(pred), torch.max(pred)))

            loss = loss_fn(pred, outputs)

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            loss.backward()
            optimizer.step()

            train_running_loss += loss.data

        model.eval()
        val_running_loss = 0.0
        for _, data in enumerate(data_loaders['val']):
            inputs = data[0]
            outputs = data[1]
            outputs = inputs

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
        print('\tEpoch: {}\tLoss: {:.4f}\tVal Loss: {:.4f}\tTime per Epoch: {}s\n'.format(
            epoch, train_loss, val_loss, time_per_epoch))

        if val_loss < current_best_validation_loss:
            print('Saving new best model with val loss: ',
                  val_loss, '\tOld Loss was: ', current_best_validation_loss)
            torch.save(model.state_dict(), path.join(
                wandb.run.dir, 'best-model.pt'))
            current_best_validation_loss = val_loss
        torch.save(model.state_dict(), path.join(
            wandb.run.dir, 'latest-model.pt'))

        if val_loss <= last_val_loss:
            early_stop_count = 0
        else:
            early_stop_count = early_stop_count + 1
        last_val_loss = val_loss

        if early_stop_count == 50:
            print('Early stopping because no val_loss improvement for 50 epochs')
            break


if __name__ == '__main__':
    main()
