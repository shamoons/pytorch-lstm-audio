import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import LSTM, GRU
from sru import SRU


class BaselineModel(nn.Module):
    def __init__(self, feature_dim=5, hidden_size=5, num_layers=2, seq_length=1, dropout=0.1):
        super(BaselineModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        # self.linear = nn.Linear(322, 161)
        # self.linear = nn.Linear(64 * seq_length, 5)
        self.lstm = nn.LSTM(input_size=feature_dim,
                            hidden_size=hidden_size, num_layers=num_layers, dropout=0.1, bidirectional=True)
        # self.lstm = SRU(input_size=feature_dim, hidden_size=hidden_size,
        #                 num_layers=num_layers, dropout=0.1, bidirectional=True, use_tanh=True, nn_rnn_compatible_return=True)
        # self.lstm = nn.GRU(input_size=feature_dim, hidden_size=feature_dim,
        #                    num_layers=num_layers, dropout=0.1, bidirectional=True)

    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = lstm_out.view(-1, lstm_out.shape[2])
        lstm_out = (lstm_out[:, :, :self.hidden_size] +
                    lstm_out[:, :, self.hidden_size:])
        # out = self.linear(lstm_out)
        # print('out', out.size())
        # out = torch.nn.functional.relu(out)
        out = torch.nn.SELU()(lstm_out)
        return out, hidden

    def init_hidden_gru(self, batch_size):
        hidden = torch.zeros(
            self.num_layers * 2, self.seq_length, self.hidden_size)
        return hidden

    def init_hidden(self, batch_size):
        hidden = torch.zeros(
            self.num_layers * 2, self.seq_length, self.hidden_size)
        cell = torch.zeros(self.num_layers * 2, self.seq_length,
                           self.hidden_size)
        return (hidden.float(), cell.float())
