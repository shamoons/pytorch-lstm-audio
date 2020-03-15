import torch


class BaselineModel(torch.nn.Module):
    def __init__(self, feature_dim=1, seq_length=32, kernel_sizes=(11, 9), strides=(1, 1)):
        super(BaselineModel, self).__init__()
        self.seq_length = seq_length

        self.conv1 = torch.nn.Conv1d(
            in_channels=feature_dim,
            out_channels=feature_dim,
            kernel_size=kernel_sizes[0],
            stride=strides[0],
            padding=kernel_sizes[0] // 2)

        self.conv2 = torch.nn.Conv1d(
            in_channels=feature_dim,
            out_channels=feature_dim,
            kernel_size=kernel_sizes[1],
            stride=strides[1],
            padding=kernel_sizes[1] // 2)

        # self.conv1 = torch.nn.Conv1d(
        #     in_channels=feature_dim,
        #     out_channels=feature_dim,
        #     kernel_size=kernel_sizes[0],
        #     stride=strides[0],
        #     padding=kernel_sizes[0] // 2)

        # self.conv2 = torch.nn.Conv1d(
        #     in_channels=feature_dim,
        #     out_channels=feature_dim,
        #     kernel_size=kernel_sizes[1],
        #     stride=strides[1],
        #     padding=kernel_sizes[1] // 2)

    def forward(self, x):
        # print('x', x.size())
        inp = x.transpose(1, 2)
        # print('inp', inp.size())

        out = self.conv1(inp)
        out = torch.nn.functional.relu(out)
        # print('out1', out.size())

        out = self.conv2(out)
        out = torch.nn.functional.relu(out)
        out = out.transpose(1, 2)
        # print('out2', out.size())

        return out


class BaselineModelLSTM(torch.nn.Module):
    def __init__(self, feature_dim=5, hidden_size=5, num_layers=2, seq_length=1, dropout=0.25):
        super(BaselineModelLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        intermediate_size = ((hidden_size * 2) + 161) // 2
        self.linear1 = torch.nn.Linear(hidden_size * 2, intermediate_size)
        self.linear2 = torch.nn.Linear(intermediate_size, 161)
        self.lstm = torch.nn.LSTM(input_size=feature_dim,
                                  hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
                                  bidirectional=True)

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
