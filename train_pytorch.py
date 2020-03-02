from baseline_model import BaselineModel
from torchsummary import summary
from audio_dataset import AudioDataset
from barbar import Bar
import torch.optim as optim
import torch
import wandb
import socket


SEQ_LENGTH = 5
BATCH_SIZE = 32
FEATURE_DIM = 161


def main():
    wandb_tags = [socket.gethostname()]
    wandb.init(project="speech-reconstruction-with-deepspeech2",
               tags=','.join(wandb_tags))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
    print(params)

    train_set = AudioDataset(
        'data/dev-noise-subtractive-250ms-1/', train_set=True, seq_length=SEQ_LENGTH, feature_dim=FEATURE_DIM)
    val_set = AudioDataset(
        'data/dev-noise-subtractive-250ms-1/', test_set=True, seq_length=SEQ_LENGTH, feature_dim=FEATURE_DIM)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, **params)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=True, **params)

    data_loaders = {'train': train_loader, 'val': val_loader}

    # print(list(train_loader)[0])
    # exit()

    model = BaselineModel(feature_dim=FEATURE_DIM,
                          hidden_size=FEATURE_DIM, seq_length=SEQ_LENGTH)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    loss_fn = torch.nn.MSELoss(reduction='sum')

    wandb.watch(model)

    for epoch in range(250):
        model.train(True)  # Set model to training mode

        train_running_loss = 0.0
        hidden = model.init_hidden(BATCH_SIZE)
        for i, data in enumerate(Bar(data_loaders['train'])):
            inputs = data[0]
            outputs = data[1]
            optimizer.zero_grad()

            pred, hidden = model(inputs, hidden)

            loss = loss_fn(pred, outputs)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            loss.backward(retain_graph=True)
            optimizer.step()

            train_running_loss += loss.data

        model.train(False)
        for i, data in enumerate(data_loaders['val']):

            inputs = data[0]
            outputs = data[1]

        epoch_loss = train_running_loss / len(data_loaders['train'])
        wandb.log({"Training Loss": epoch_loss})
        print('\tEpoch: {}\tLoss: {:.4f}'.format(epoch, epoch_loss))
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))


if __name__ == '__main__':
    main()
