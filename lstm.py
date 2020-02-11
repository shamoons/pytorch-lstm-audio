import torch.nn as nn


class MyLSTM(nn.Module):
    def __init__(self):
        super(MyLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=4000, hidden_size=4000,
                            num_layers=2)

    def forward(self, x):
        y = self.lstm(x)
        return y
