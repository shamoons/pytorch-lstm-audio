import torch
import torch.nn as nn
from torch.nn import LSTM, GRU


class BaselineModel(nn.Module):
    def __init__(self, feature_dim=5, hidden_size=5, num_layers=2, seq_length=1, dropout=0.25):
        super(BaselineModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        intermediate_size = ((hidden_size * 2) + 161) // 2
        self.linear1 = nn.Linear(hidden_size * 2, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, 161)
        self.lstm = nn.LSTM(input_size=feature_dim,
                            hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=True)

    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)
        batch_size = lstm_out.size(0)

        flattened_out = lstm_out.view(-1, self.hidden_size * 2)

        out = self.linear1(flattened_out)

        out = torch.nn.functional.relu(out)
        out = self.linear2(out)
        out = torch.nn.functional.relu(out)

        out = out.view(batch_size, self.seq_length, -1)

        return out, hidden

    def init_hidden_gru(self):
        hidden = torch.zeros(
            self.num_layers * 2, self.seq_length, self.hidden_size)
        return hidden

    def init_hidden(self, batch_size):
        hidden = torch.zeros(
            self.num_layers * 2, self.seq_length, self.hidden_size)
        cell = torch.zeros(self.num_layers * 2, self.seq_length,
                           self.hidden_size)
        return (hidden.float(), cell.float())
