import torch.nn as nn
from torch.autograd import Variable


class BaselineModel(nn.Module):
    def __init__(self, feature_dim=5, hidden_size=5, num_layers=2, batch_size=32):
        super(BaselineModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=feature_dim,
                            hidden_size=hidden_size, num_layers=num_layers)

    def forward(self, x, hidden):
        lstm_out, hidden = self.lstm(x, hidden)
        return lstm_out, hidden

    def init_hidden(self, batch_size):
        hidden = Variable(next(self.parameters()).data.new(
            self.num_layers, batch_size, self.hidden_size))
        cell = Variable(next(self.parameters()).data.new(
            self.num_layers, batch_size, self.hidden_size))
        return (hidden, cell)
