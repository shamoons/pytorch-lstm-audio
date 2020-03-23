import torch


class BaselineModel(torch.nn.Module):
    def __init__(self, feature_dim=1, kernel_sizes=(15, 11, 9, 7, 5), make_4d=False, dropout=0.01, initialize_weights= True):
        super(BaselineModel, self).__init__()
        self.make_4d = make_4d

        self.dropout = torch.nn.Dropout(p=0.25)

        conv1_out_channels = int(feature_dim * 0.95)
        conv2_out_channels = int(conv1_out_channels * 0.85)
        conv3_out_channels = int(conv2_out_channels * 0.75)
        conv4_out_channels = int(conv3_out_channels * 1.25)
        conv5_out_channels = feature_dim
        # conv1_out_channels = conv2_out_channels = conv3_out_channels = conv4_out_channels = feature_dim

        self.conv1 = torch.nn.Conv1d(
            in_channels=feature_dim,
            out_channels=conv1_out_channels,
            kernel_size=kernel_sizes[0],
            stride=1,
            padding=kernel_sizes[0] // 2)


        self.conv2 = torch.nn.Conv1d(
            in_channels=conv1_out_channels,
            out_channels=conv2_out_channels,
            kernel_size=kernel_sizes[1],
            stride=1,
            padding=kernel_sizes[1] // 2)

        self.conv3 = torch.nn.Conv1d(
            in_channels=conv2_out_channels,
            out_channels=conv3_out_channels,
            kernel_size=kernel_sizes[2],
            stride=1,
            padding=kernel_sizes[2] // 2)

        self.conv4 = torch.nn.Conv1d(
            in_channels=conv3_out_channels,
            out_channels=conv4_out_channels,
            kernel_size=kernel_sizes[3],
            stride=1,
            padding=kernel_sizes[3] // 2)

        self.conv5 = torch.nn.Conv1d(
            in_channels=conv4_out_channels,
            out_channels=conv5_out_channels,
            kernel_size=kernel_sizes[4],
            stride=1,
            padding=kernel_sizes[4] // 2)

        if initialize_weights:
            torch.nn.init.zeros_(self.conv1.weight)
            torch.nn.init.zeros_(self.conv2.weight)
            torch.nn.init.zeros_(self.conv3.weight)
            torch.nn.init.zeros_(self.conv4.weight)
            torch.nn.init.zeros_(self.conv5.weight)

            self.conv1.weight.data[:, :, 5] = torch.eye(conv1_out_channels, feature_dim)
            self.conv2.weight.data[:, :, 4] = torch.eye(conv2_out_channels, conv1_out_channels)
            self.conv3.weight.data[:, :, 3] = torch.eye(conv3_out_channels, conv2_out_channels)
            self.conv4.weight.data[:, :, 2] = torch.eye(conv4_out_channels, conv3_out_channels)
            self.conv5.weight.data[:, :, 1] = torch.eye(conv5_out_channels, conv4_out_channels)
            # self.conv1.weight.data[:, :, 5] = 1.0
            # self.conv2.weight.data[:, :, 4] = 1.0
            # self.conv3.weight.data[:, :, 3] = 1.0
            # self.conv4.weight.data[:, :, 2] = 1.0
            # self.conv5.weight.data[:, :, 1] = 1.0

            torch.nn.init.zeros_(self.conv1.bias)
            torch.nn.init.zeros_(self.conv2.bias)
            torch.nn.init.zeros_(self.conv3.bias)
            torch.nn.init.zeros_(self.conv4.bias)
            torch.nn.init.zeros_(self.conv5.bias)

        self.selu = torch.nn.SELU()
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU6()
        # print(self.conv4.weight.data)
        # print(self.conv4.bias.data)
        

    def forward(self, x):
        if self.make_4d:
            x = x.view(x.size(0), x.size(3), x.size(2))

        
        inp = x.transpose(1, 2)
        # print('\ninp', inp.min(), inp.mean(), inp.max(), inp.size())
        
        out = self.conv1(inp)
        out = self.tanh(out)
        # out = self.dropout(out)
        # print('\nout1', out.min(), out.mean(), out.max(), out.size())
        

        out = self.conv2(out)
        out = self.tanh(out)
        # out = self.dropout(out)
        # print('\nout2', out.min(), out.mean(), out.max(), out.size())
        # quit()

        out = self.conv3(out)
        out = self.tanh(out)
        # out = self.dropout(out)
        # print('\nout3', out.min(), out.mean(), out.max(), out.size())

        out = self.conv4(out)
        out = self.selu(out)
        # out = self.dropout(out)
        # print('\nout4', out.min(), out.mean(), out.max(), out.size())


        out = self.conv5(out)
        out = self.relu(out)
        # out = self.dropout(out)
        # print('\nout5', out.min(), out.mean(), out.max(), out.size())

        out = out.transpose(1, 2)
        if self.make_4d:
            out = out.reshape(out.size(0), 1, out.size(2), out.size(1))


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
