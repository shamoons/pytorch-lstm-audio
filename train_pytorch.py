from audio_dataset import AudioDataset
from barbar import Bar
from baseline_model import BaselineModel
from torchsummary import summary
import argparse
import os.path as path
import socket
import torch
import torch.optim as optim
import wandb
import time


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--audio_path', help='Path for corrupted audio',
                        default='data/dev-noise-subtractive-250ms-1')

    parser.add_argument(
        "--hidden_size", help="Hidden size for the LSTM Layers", type=int, default=128)

    parser.add_argument(
        "--num_layers", help="Number of layers in the model", type=int, default=2)

    parser.add_argument('--seq_length', help='Length of sequences of the spectrogram',
                        type=int, default=20)
    parser.add_argument('--feature_dim', help='Feature dimension',
                        type=int, default=161)

    parser.add_argument('--epochs', help='Epochs to run',
                        type=int, default=250)

    parser.add_argument('--batch_size', help='Batch size',
                        type=int, default=64)

    parser.add_argument('--repeat_sample', help='How many times to sample each file',
                        type=int, default=1)

    args = parser.parse_args()

    return args


def initialize():
    torch.set_default_tensor_type('torch.FloatTensor')
    wandb_tags = [socket.gethostname()]
    wandb.init(project="speech-reconstruction-with-deepspeech2",
               tags=','.join(wandb_tags))
    wandb.save('*.pt')


def main():
    initialize()
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}

    train_set = AudioDataset(
        args.audio_path, train_set=True, seq_length=args.seq_length, feature_dim=args.feature_dim, repeat_sample=args.repeat_sample)
    val_set = AudioDataset(
        args.audio_path, test_set=True, seq_length=args.seq_length, feature_dim=args.feature_dim)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=10, **params)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=True, num_workers=10, **params)

    data_loaders = {'train': train_loader, 'val': val_loader}

    model = BaselineModel(feature_dim=args.feature_dim,
                          hidden_size=args.feature_dim, seq_length=args.seq_length, num_layers=args.num_layers)
    # optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    scheduler = optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=0.0001, max_lr=0.01, mode='triangular2', step_size_up=args.epochs // 8, step_size_down=args.epochs // 4)

    loss_fn = torch.nn.MSELoss(reduction='sum')

    wandb.watch(model)

    current_best_validation_loss = 10000
    model = model.float()

    if(torch.cuda.is_available()):
        model.cuda()

    for epoch in range(args.epochs):
        model.train(True)  # Set model to training mode
        start_time = time.time()
        train_running_loss = 0.0
        for _, data in enumerate(Bar(data_loaders['train'])):
            # Keeping the hidden re-init here because each iteration is a batch of sequences
            # and batches are unrelated
            hidden = model.init_hidden(args.batch_size)

            inputs = data[0]
            outputs = data[1]
            if(torch.cuda.is_available()):
                inputs = inputs.cuda()
                outputs = outputs.cuda()
                hidden = hidden[0].cuda(), hidden[1].cuda()

            optimizer.zero_grad()

            pred, hidden = model(inputs, hidden)

            loss = loss_fn(pred, outputs)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            loss.backward(retain_graph=True)
            optimizer.step()

            train_running_loss += loss.data

        scheduler.step()

        model.eval()
        val_running_loss = 0.0
        for _, data in enumerate(data_loaders['val']):
            inputs = data[0]
            outputs = data[1]

            if(torch.cuda.is_available()):
                inputs = inputs.cuda()
                outputs = outputs.cuda()

            pred, _ = model(inputs)

            loss = loss_fn(pred, outputs)
            val_running_loss += loss.data

        time_per_epoch = int(time.time() - start_time)
        train_loss = train_running_loss / len(data_loaders['train'])
        val_loss = val_running_loss / len(data_loaders['val'])
        wandb.log({
            "Training Loss": train_loss,
            'Validation Loss': val_loss,
            'Epoch': epoch,
            'Time per Epoch': time_per_epoch,
            'Learning Rate': scheduler.get_last_lr()[0]
        })
        print('\tEpoch: {}\tLoss: {:.4f}\tVal Loss: {:.4f}\tTime per Epoch: {}s\n'.format(
            epoch, train_loss, val_loss, time_per_epoch))

        if val_loss < current_best_validation_loss:
            torch.save(model, path.join(wandb.run.dir, 'model.pt'))
            current_best_validation_loss = val_loss


if __name__ == '__main__':
    main()
