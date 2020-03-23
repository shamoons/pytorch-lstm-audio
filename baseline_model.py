import torch


class BaselineModel(torch.nn.Module):
    def __init__(self, feature_dim, kernel_size, kernel_size_step, final_kernel_size, make_4d=False, dropout=0.01, verbose=False):
        super(BaselineModel, self).__init__()
        self.make_4d = make_4d
        self.verbose = verbose

        self.dropout = torch.nn.Dropout(p=dropout)

        conv5_out_channels = feature_dim
        conv1_out_channels = conv2_out_channels = conv3_out_channels = conv4_out_channels = conv5_out_channels
        kernel_sizes = [kernel_size, kernel_size + kernel_size_step, kernel_size + 2 *
                        kernel_size_step, kernel_size + 3 * kernel_size_step, kernel_size + 4 * kernel_size_step]


        self.conv1 = self.conv_layer(in_channels=feature_dim, kernel_size=kernel_sizes[0])
        self.conv2 = self.conv_layer(in_channels=feature_dim, kernel_size=kernel_sizes[1])
        self.conv3 = self.conv_layer(in_channels=feature_dim, kernel_size=kernel_sizes[2])
        self.conv4 = self.conv_layer(in_channels=feature_dim, kernel_size=kernel_sizes[3])
        self.conv5 = self.conv_layer(in_channels=feature_dim, kernel_size=kernel_sizes[4])
        self.final_conv = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=(feature_dim // 8) * 5,
                out_channels=feature_dim // 4,
                kernel_size=final_kernel_size,
                stride=1,
                padding=final_kernel_size // 2
            ),
            torch.nn.PReLU(num_parameters=feature_dim // 4),
            torch.nn.Conv1d(
                in_channels=feature_dim // 4,
                out_channels=feature_dim // 2,
                kernel_size=final_kernel_size,
                stride=1,
                padding=final_kernel_size // 2
            ),
            torch.nn.PReLU(num_parameters=feature_dim // 2),
            torch.nn.Conv1d(
                in_channels=feature_dim // 2,
                out_channels=feature_dim,
                kernel_size=final_kernel_size,
                stride=1,
                padding=final_kernel_size // 2
            ),
            torch.nn.ReLU6()
        )

        return
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=feature_dim,
                out_channels=conv1_out_channels // 2,
                kernel_size=kernel_sizes[0],
                stride=1,
                padding=kernel_sizes[0] // 2
            ),
            torch.nn.PReLU(num_parameters=conv1_out_channels // 2),
            torch.nn.Conv1d(
                in_channels=conv1_out_channels // 2,
                out_channels=conv1_out_channels // 4,
                kernel_size=kernel_sizes[0],
                stride=1,
                padding=kernel_sizes[0] // 2
            ),
            torch.nn.PReLU(num_parameters=conv1_out_channels // 4),
            torch.nn.Conv1d(
                in_channels=conv1_out_channels // 4,
                out_channels=conv1_out_channels // 5,
                kernel_size=kernel_sizes[0],
                stride=1,
                padding=kernel_sizes[0] // 2
            ),
            torch.nn.PReLU(num_parameters=conv1_out_channels // 4)
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=conv1_out_channels,
                out_channels=conv2_out_channels // 2,
                kernel_size=kernel_sizes[1],
                stride=1,
                padding=kernel_sizes[1] // 2
            ),
            torch.nn.PReLU(num_parameters=conv2_out_channels // 2),
            torch.nn.Conv1d(
                in_channels=conv2_out_channels // 2,
                out_channels=conv2_out_channels // 4,
                kernel_size=kernel_sizes[1],
                stride=1,
                padding=kernel_sizes[1] // 2
            ),
            torch.nn.PReLU(num_parameters=conv2_out_channels // 4)
        )

        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=conv2_out_channels,
                out_channels=conv3_out_channels // 2,
                kernel_size=kernel_sizes[2],
                stride=1,
                padding=kernel_sizes[2] // 2
            ),
            torch.nn.PReLU(num_parameters=conv3_out_channels // 2),
            torch.nn.Conv1d(
                in_channels=conv3_out_channels // 2,
                out_channels=conv3_out_channels // 4,
                kernel_size=kernel_sizes[2],
                stride=1,
                padding=kernel_sizes[2] // 2
            ),
            torch.nn.PReLU(num_parameters=conv3_out_channels // 4)
        )

        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=conv3_out_channels,
                out_channels=conv4_out_channels // 2,
                kernel_size=kernel_sizes[3],
                stride=1,
                padding=kernel_sizes[3] // 2
            ),
            torch.nn.PReLU(num_parameters=conv4_out_channels // 2),
            torch.nn.Conv1d(
                in_channels=conv4_out_channels // 2,
                out_channels=conv4_out_channels // 4,
                kernel_size=kernel_sizes[3],
                stride=1,
                padding=kernel_sizes[3] // 2
            ),
            torch.nn.PReLU(num_parameters=conv4_out_channels // 4)
        )

        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=conv4_out_channels,
                out_channels=conv5_out_channels // 2,
                kernel_size=kernel_sizes[4],
                stride=1,
                padding=kernel_sizes[4] // 2
            ),
            torch.nn.PReLU(num_parameters=conv5_out_channels // 2),
            torch.nn.Conv1d(
                in_channels=conv5_out_channels // 2,
                out_channels=conv5_out_channels // 4,
                kernel_size=kernel_sizes[4],
                stride=1,
                padding=kernel_sizes[4] // 2
            ),
            torch.nn.PReLU(num_parameters=conv5_out_channels // 4)
        )

        self.final_conv = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=(feature_dim // 4) * 5,
                out_channels=feature_dim // 2,
                kernel_size=final_kernel_size,
                stride=1,
                padding=final_kernel_size // 2
            ),
            torch.nn.PReLU(num_parameters=feature_dim // 2),
            torch.nn.Conv1d(
                in_channels=feature_dim // 2,
                out_channels=feature_dim,
                kernel_size=final_kernel_size,
                stride=1,
                padding=final_kernel_size // 2
            ),
            torch.nn.ReLU6()
        )

    def conv_layer(self, in_channels, kernel_size):
        return torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
            torch.nn.PReLU(num_parameters=in_channels // 2),
            torch.nn.Conv1d(
                in_channels=in_channels // 2,
                out_channels=in_channels // 4,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
            torch.nn.PReLU(num_parameters=in_channels // 4),
            torch.nn.Conv1d(
                in_channels=in_channels // 4,
                out_channels=in_channels // 8,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
            torch.nn.PReLU(num_parameters=in_channels // 8)
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

        # quit()
        out = out.transpose(1, 2)
        if self.make_4d:
            out = out.reshape(out.size(0), 1, out.size(2), out.size(1))

        return out
