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
    loss_fn = torch.nn.MSELoss(reduction='mean')

    for epoch in range(300):
        for i, data in enumerate(train_loader):
            inputs = data[0]
            outputs = data[1]

            print('inputs', inputs, inputs.size())
            print('outputs', outputs, outputs.size())
            optimizer.zero_grad()

            pred = model(inputs)
            print('pred', pred[0], pred[0].size())

            loss = loss_fn(pred[0], outputs)

            model.zero_grad()

            loss.backward()
            optimizer.step()

            print('Epoch: ', epoch, '\ti: ', i, '\tLoss: ', loss)

            print('\n')


if __name__ == '__main__':
    main()
