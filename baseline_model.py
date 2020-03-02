import torch
import torch.nn as nn
from torch.autograd import Variable


class BaselineModel(nn.Module):
    def __init__(self, feature_dim=5, hidden_size=5, num_layers=2, seq_length=1):
        super(BaselineModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.seq_length = seq_length

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
        hidden = torch.zeros(
            self.num_layers, self.seq_length, self.hidden_size)
        cell = torch.zeros(self.num_layers, self.seq_length, self.hidden_size)
        # print('hidden', hidden.size())
        # print('cell', cell.size())
        return (hidden, cell)
