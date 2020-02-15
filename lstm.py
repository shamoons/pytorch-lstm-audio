import torch.nn as nn


class MyLSTMEncoder(nn.Module):
    def __init__(self, input_size=4000):
        super(MyLSTMEncoder, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=input_size,
                            num_layers=2)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        # y = self.lstm(x)
        return hidden, cell


class MyLSTMDecoder(nn.Module):
    def __init__(self):
        super(MyLSTMDecoder, self).__init__()


class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()

        self.encoder = MyLSTMEncoder()
        self.decoder = MyLSTMDecoder()
