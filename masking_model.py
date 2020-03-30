import torch
import numpy as np


class MaskingModel(torch.nn.Module):
    def __init__(self, feature_dim, kernel_size, kernel_size_step, final_kernel_size, device, make_4d=False, dropout=0.01, verbose=False):
        super(MaskingModel, self).__init__()
        self.make_4d = make_4d
        self.verbose = verbose
        self.device = device

        self.dropout = torch.nn.Dropout(p=dropout)

        kernel_sizes = [kernel_size, kernel_size + kernel_size_step, kernel_size + 2 *
                        kernel_size_step, kernel_size + 3 * kernel_size_step, kernel_size + 4 * kernel_size_step]

        self.conv1 = self.conv_layer(in_channels=feature_dim, kernel_size=kernel_sizes[0])
        self.conv2 = self.conv_layer(in_channels=feature_dim, kernel_size=kernel_sizes[1])
        self.conv3 = self.conv_layer(in_channels=feature_dim, kernel_size=kernel_sizes[2])
        self.conv4 = self.conv_layer(in_channels=feature_dim, kernel_size=kernel_sizes[3])
        self.conv5 = self.conv_layer(in_channels=feature_dim, kernel_size=kernel_sizes[4])
        self.final_conv = torch.nn.Sequential(
            # torch.nn.Conv1d(
            #     in_channels=(feature_dim // 8) * 5,
            #     out_channels=(feature_dim // 8) * 5,
            #     kernel_size=final_kernel_size,
            #     stride=1,
            #     padding=final_kernel_size // 2,
            #     groups=(feature_dim // 8) * 5
            # ),
            # torch.nn.PReLU(num_parameters=(feature_dim // 8) * 5),
            torch.nn.Conv1d(
                in_channels=(feature_dim // 8) * 5,
                out_channels=feature_dim // 16,
                kernel_size=final_kernel_size,
                stride=1,
                padding=final_kernel_size // 2
            ),
            torch.nn.ReLU(),
            # torch.nn.PReLU(num_parameters=feature_dim // 16),
            torch.nn.Conv1d(
                in_channels=feature_dim // 16,
                out_channels=feature_dim // 32,
                kernel_size=final_kernel_size,
                stride=1,
                padding=final_kernel_size // 2
            ),
            torch.nn.ReLU(),
            # torch.nn.PReLU(num_parameters=feature_dim // 32),
            torch.nn.Conv1d(
                in_channels=feature_dim // 32,
                out_channels=1,
                kernel_size=final_kernel_size,
                stride=1,
                padding=final_kernel_size // 2
            )
        )


    def conv_layer(self, in_channels, kernel_size):
        return torch.nn.Sequential(
            # torch.nn.Conv1d(
            #     in_channels=in_channels,
            #     out_channels=in_channels,
            #     kernel_size=kernel_size,
            #     stride=1,
            #     padding=kernel_size // 2,
            #     groups=in_channels
            # ),
            # torch.nn.PReLU(num_parameters=in_channels),
            torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
            torch.nn.ReLU(),
            # torch.nn.PReLU(num_parameters=in_channels // 2),
            torch.nn.Conv1d(
                in_channels=in_channels // 2,
                out_channels=in_channels // 4,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
            torch.nn.ReLU(),
            # torch.nn.PReLU(num_parameters=in_channels // 4),
            torch.nn.Conv1d(
                in_channels=in_channels // 4,
                out_channels=in_channels // 8,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
            torch.nn.ReLU()
            # torch.nn.PReLU(num_parameters=in_channels // 8)
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

        # out = torch.cat((out1, out2, out3, out4, out5), dim=1)
        stacked = torch.stack((out1, out2, out3, out4, out5), dim=2)
        out = torch.flatten(stacked, start_dim=1, end_dim=2)

        out = self.final_conv(out)
        out = out.view(out.size(0), out.size(2))

        if self.verbose:
            print('\nout\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
                torch.mean(out), torch.std(out), torch.min(out), torch.max(out), out.size()))
        

        if self.make_4d:
            out = out.reshape(out.size(0), 1, out.size(2), out.size(1))

        return out

    def expand_mask(self, mask, seq_length, multiple=3):
        expanded_mask = []

        for _, m in enumerate(mask):
            # print('m', m, (m != 0).nonzero())
            nonzero = (m != 0).nonzero()
            # print(m)
            # print('nonzero', nonzero)
            # print(nonzero[0])
            # print(nonzero[-1])
            # print('before')

            new_mask = np.zeros(m.size())
            if nonzero.size(0) > 0:
                # print('here')
                start_1 = (m != 0).nonzero()[0][0]
                end_1 = (m != 0).nonzero()[-1][0]
                mask_length = end_1 - start_1
                new_mask[max(0, start_1 - mask_length):min(end_1 + mask_length, seq_length)] = 1

            expanded_mask.append(new_mask)

        expanded_mask = torch.Tensor(expanded_mask, device=self.device)

        return expanded_mask
