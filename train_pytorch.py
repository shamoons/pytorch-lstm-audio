import torch
import torch.optim as optim
from audio_dataset import AudioDataset
from torchsummary import summary
from baseline_model import BaselineModel


SEQ_LENGTH = 13
BATCH_SIZE = 7


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
    print(params)

    train_set = AudioDataset(
        'data/dev-noise-subtractive-250ms-1/', train_set=True, seq_length=SEQ_LENGTH, batch_size=BATCH_SIZE)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, **params)

    model = BaselineModel(batch_size=BATCH_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    loss_fn = torch.nn.MSELoss(reduction='sum')

    for epoch in range(250):
        hidden = model.init_hidden(13)
        # hidden = (torch.zeros(2, 13, 5),
        #           torch.zeros(2, 13, 5))
        # model.hidden = hidden
        for i, data in enumerate(train_loader):
            inputs = data[0]
            outputs = data[1]

            print('inputs',  inputs.size())
            # print('outputs', outputs.size())

            optimizer.zero_grad()
            model.zero_grad()

            # print('inputs', inputs)
            pred, hidden = model(inputs, hidden)

            loss = loss_fn(pred[0], outputs)

            loss.backward(retain_graph=True)
            optimizer.step()

            print('Epoch: ', epoch, '\ti: ', i, '\tLoss: ', loss)


def main2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = AudioDataset(
        'data/dev-noise-subtractive-250ms-1/', train_set=True)

    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=1, shuffle=True, **kwargs)

    model = MyLSTMEncoder(input_size=5)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    loss_fn = torch.nn.MSELoss(reduction='sum')

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

            if i % 10 == 0:
                print('Epoch: ', epoch, '\ti: ', i, '\tLoss: ', loss)

            print('\n')
    print('Final Loss: ', loss)


if __name__ == '__main__':
    main()
