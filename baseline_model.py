import torch


class BaselineModel(torch.nn.Module):
    def __init__(self, feature_dim=1, kernel_sizes=(21, 17, 13, 9, 5), final_kernel_size=11, make_4d=False, dropout=0.01, verbose=False):
        super(BaselineModel, self).__init__()
        self.make_4d = make_4d
        self.verbose = verbose

        self.dropout = torch.nn.Dropout(p=dropout)

        # conv1_out_channels = out_channels[0]
        # conv2_out_channels = out_channels[1]
        # conv3_out_channels = out_channels[2]
        # conv4_out_channels = out_channels[3]
        conv5_out_channels = feature_dim
        conv1_out_channels = conv2_out_channels = conv3_out_channels = conv4_out_channels = conv5_out_channels

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=feature_dim,
                out_channels=conv1_out_channels // 2,
                kernel_size=kernel_sizes[0],
                stride=1,
                padding=kernel_sizes[0] // 2
            ),
            torch.nn.ReLU6(),
            torch.nn.Conv1d(
                in_channels=conv1_out_channels // 2,
                out_channels=conv1_out_channels // 4,
                kernel_size=kernel_sizes[0],
                stride=1,
                padding=kernel_sizes[0] // 2
            ),
            torch.nn.ReLU6()
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=conv1_out_channels,
                out_channels=conv2_out_channels // 2,
                kernel_size=kernel_sizes[1],
                stride=1,
                padding=kernel_sizes[1] // 2
            ),
            torch.nn.ReLU6(),
            torch.nn.Conv1d(
                in_channels=conv2_out_channels // 2,
                out_channels=conv2_out_channels // 4,
                kernel_size=kernel_sizes[1],
                stride=1,
                padding=kernel_sizes[1] // 2
            ),
            torch.nn.ReLU6()
        )

        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=conv2_out_channels,
                out_channels=conv3_out_channels // 2,
                kernel_size=kernel_sizes[2],
                stride=1,
                padding=kernel_sizes[2] // 2
            ),
            torch.nn.ReLU6(),
            torch.nn.Conv1d(
                in_channels=conv3_out_channels // 2,
                out_channels=conv3_out_channels // 4,
                kernel_size=kernel_sizes[2],
                stride=1,
                padding=kernel_sizes[2] // 2
            ),
            torch.nn.ReLU6()
        )

        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=conv3_out_channels,
                out_channels=conv4_out_channels // 2,
                kernel_size=kernel_sizes[3],
                stride=1,
                padding=kernel_sizes[3] // 2
            ),
            torch.nn.ReLU6(),
            torch.nn.Conv1d(
                in_channels=conv4_out_channels // 2,
                out_channels=conv4_out_channels // 4,
                kernel_size=kernel_sizes[3],
                stride=1,
                padding=kernel_sizes[3] // 2
            ),
            torch.nn.ReLU6()
        )

        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=conv4_out_channels,
                out_channels=conv5_out_channels // 2,
                kernel_size=kernel_sizes[4],
                stride=1,
                padding=kernel_sizes[4] // 2
            ),
            torch.nn.ReLU6(),
            torch.nn.Conv1d(
                in_channels=conv5_out_channels // 2,
                out_channels=conv5_out_channels // 4,
                kernel_size=kernel_sizes[4],
                stride=1,
                padding=kernel_sizes[4] // 2
            ),
            torch.nn.ReLU6()
        )

        self.final_conv = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=(feature_dim // 4) * 5,
                out_channels=feature_dim,
                kernel_size=final_kernel_size,
                stride=1,
                padding=final_kernel_size // 2
            ),
            torch.nn.ReLU6()
        )

    def forward(self, x):
        if self.make_4d:
            x = x.view(x.size(0), x.size(3), x.size(2))

        inp = x.transpose(1, 2)
        if self.verbose:
            print('\ninp\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
                torch.mean(inp), torch.std(inp), torch.min(inp), torch.max(inp), inp.size()))

        out1 = self.conv1(inp)
        if self.verbose:
            print('\nout1\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
                torch.mean(out1), torch.std(out1), torch.min(out1), torch.max(out1), out1.size()))

        out2 = self.conv2(inp)
        if self.verbose:
            print('\nout2\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
                torch.mean(out2), torch.std(out2), torch.min(out2), torch.max(out2), out2.size()))

        out3 = self.conv3(inp)
        if self.verbose:
            print('\nout3\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
                torch.mean(out3), torch.std(out3), torch.min(out3), torch.max(out3), out3.size()))

        out4 = self.conv4(inp)
        if self.verbose:
            print('\nout4\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
                torch.mean(out4), torch.std(out4), torch.min(out4), torch.max(out4), out4.size()))


        out5 = self.conv5(inp)
        if self.verbose:
            print('\nout5\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
                torch.mean(out5), torch.std(out5), torch.min(out5), torch.max(out5), out5.size()))


        out = torch.cat((out1, out2, out3, out4, out5), dim=1)
        out = self.final_conv(out)
        if self.verbose:
            print('\nout\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
                torch.mean(out), torch.std(out), torch.min(out), torch.max(out), out.size()))


        out = out.transpose(1, 2)
        if self.make_4d:
            out = out.reshape(out.size(0), 1, out.size(2), out.size(1))

        return out

    def forward_old(self, x):
        if self.make_4d:
            x = x.view(x.size(0), x.size(3), x.size(2))

        inp = x.transpose(1, 2)
        if self.verbose == True:
            print('\ninp', inp.min(), inp.mean(), inp.max(), inp.size())

        out = self.conv1(inp)
        out = self.tanh(out)
        # out = self.dropout(out)
        if self.verbose == True:
            print('\nout1', out.min(), out.mean(), out.max(), out.size())

        out = self.conv2(out)
        out = self.tanh(out)
        # out = self.dropout(out)
        if self.verbose == True:
            print('\nout2', out.min(), out.mean(), out.max(), out.size())
        # quit()

        out = self.conv3(out)
        out = self.tanh(out)
        # out = self.dropout(out)
        if self.verbose == True:
            print('\nout3', out.min(), out.mean(), out.max(), out.size())

        out = self.conv4(out)
        out = self.selu(out)
        # out = self.dropout(out)
        if self.verbose == True:
            print('\nout4', out.min(), out.mean(), out.max(), out.size())

        out = self.conv5(out)
        out = self.relu(out)
        # out = self.dropout(out)
        if self.verbose == True:
            print('\nout5', out.min(), out.mean(), out.max(), out.size())

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
