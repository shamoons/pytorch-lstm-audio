import torch
import torch.optim as optim
from audiodataset import AudioDataset
from lstm import MyLSTM


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = AudioDataset(
        'data/dev-noise-subtractive-250ms-1/', train_set=True)

    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=1, shuffle=True, **kwargs)

    model = MyLSTM()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    loss_fn = torch.nn.MSELoss(reduction='sum')

    for epoch in range(300):
        for inputs, outputs in train_loader:
            optimizer.zero_grad()

            print('inputs', inputs.size(), inputs.unsqueeze(1).size())
            print('outputs', outputs.size(), outputs.unsqueeze(0).size())

            pred = model(inputs.unsqueeze(1))

            loss = loss_fn(pred[0], outputs.unsqueeze(0))

            model.zero_grad()

            loss.backward()
            optimizer.step()

            print('loss', loss)

            print('\n')


if __name__ == '__main__':
    main()
