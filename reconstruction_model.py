import torch


class ReconstructionModel(torch.nn.Module):
    def __init__(self, feature_dim=161, kernel_size=25, kernel_size_step=-4, make_4d=False, dropout=0.01, side_length=320, verbose=False):
        super(ReconstructionModel, self).__init__()
        self.make_4d = make_4d
        self.verbose = verbose
        self.side_length = side_length

        self.dropout = torch.nn.Dropout(p=dropout)

        kernel_sizes = [kernel_size, kernel_size + kernel_size_step, kernel_size + 2 *
                        kernel_size_step, kernel_size + 3 * kernel_size_step, kernel_size + 4 * kernel_size_step, kernel_size + 5 * kernel_size_step]

        self.linear_left = torch.nn.Sequential(
            torch.nn.Linear(20, (feature_dim + 20) // 2),
            torch.nn.ReLU(),
            torch.nn.Linear((feature_dim + 20) // 2, feature_dim),
            torch.nn.ReLU6()
        )

        self.linear_right = torch.nn.Sequential(
            torch.nn.Linear(20, (feature_dim + 20) // 2),
            torch.nn.ReLU(),
            torch.nn.Linear((feature_dim + 20) // 2, feature_dim),
            torch.nn.ReLU6()
        )

        self.left_conv1 = self.conv_layer(
            in_channels=feature_dim, kernel_size=kernel_sizes[0])
        self.left_conv2 = self.conv_layer(
            in_channels=feature_dim, kernel_size=kernel_sizes[1])
        self.left_conv3 = self.conv_layer(
            in_channels=feature_dim, kernel_size=kernel_sizes[2])
        self.left_conv4 = self.conv_layer(
            in_channels=feature_dim, kernel_size=kernel_sizes[3])
        self.left_conv5 = self.conv_layer(
            in_channels=feature_dim, kernel_size=kernel_sizes[4])
        self.left_conv6 = self.conv_layer(
            in_channels=feature_dim, kernel_size=kernel_sizes[5])

        self.right_conv1 = self.conv_layer(
            in_channels=feature_dim, kernel_size=kernel_sizes[0])
        self.right_conv2 = self.conv_layer(
            in_channels=feature_dim, kernel_size=kernel_sizes[1])
        self.right_conv3 = self.conv_layer(
            in_channels=feature_dim, kernel_size=kernel_sizes[2])
        self.right_conv4 = self.conv_layer(
            in_channels=feature_dim, kernel_size=kernel_sizes[3])
        self.right_conv5 = self.conv_layer(
            in_channels=feature_dim, kernel_size=kernel_sizes[4])
        self.right_conv6 = self.conv_layer(
            in_channels=feature_dim, kernel_size=kernel_sizes[5])

        self.left_downscale_time_conv = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=feature_dim // 8,
                out_channels=feature_dim // 8,
                kernel_size=2,
                stride=1,
                dilation=1,
                padding=0
            ),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                in_channels=feature_dim // 8,
                out_channels=feature_dim // 8,
                kernel_size=2,
                stride=1,
                dilation=2,
                padding=0
            ),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                in_channels=feature_dim // 8,
                out_channels=feature_dim // 8,
                kernel_size=2,
                stride=1,
                dilation=4,
                padding=0
            ),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                in_channels=feature_dim // 8,
                out_channels=feature_dim // 8,
                kernel_size=2,
                stride=1,
                dilation=8,
                padding=0
            ),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                in_channels=feature_dim // 8,
                out_channels=feature_dim // 8,
                kernel_size=2,
                stride=1,
                dilation=16,
                padding=0
            ),
            torch.nn.ReLU()
        )

        self.right_downscale_time_conv = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=feature_dim // 8,
                out_channels=feature_dim // 8,
                kernel_size=11,
                stride=1,
                dilation=1,
                padding=0
            ),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                in_channels=feature_dim // 8,
                out_channels=feature_dim // 8,
                kernel_size=11,
                stride=1,
                dilation=2,
                padding=0
            ),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                in_channels=feature_dim // 8,
                out_channels=feature_dim // 8,
                kernel_size=11,
                stride=1,
                dilation=4,
                padding=0
            ),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                in_channels=feature_dim // 8,
                out_channels=feature_dim // 8,
                kernel_size=11,
                stride=1,
                dilation=8,
                padding=0
            ),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                in_channels=feature_dim // 8,
                out_channels=feature_dim // 8,
                kernel_size=11,
                stride=1,
                dilation=16,
                padding=0
            ),
            torch.nn.ReLU()
        )

    def conv_layer(self, in_channels, kernel_size):
        return torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                groups=in_channels
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(in_channels),
            torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                in_channels=in_channels // 2,
                out_channels=in_channels // 4,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                in_channels=in_channels // 4,
                out_channels=in_channels // 8,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
            torch.nn.ReLU()
        )

    def forward(self, x, mask):
        if self.make_4d:
            x = x.view(x.size(0), x.size(3), x.size(2))

        inp = x

        mask_sums = torch.sum(mask, 1)

        max_mask_len = torch.max(mask_sums).int()

        outputs = torch.zeros((inp.size(0), max_mask_len, inp.size(2))).to(x.device)
        # print(f"sums: {mask_sums}")
        # print(
        #     f"mask size: {mask.size()}\tmask sum: {mask_sums.size()}\tmax_mask_len: {max_mask_len}\tinp: {inp.size()}\toutputs: {outputs.size()}")

        left_input = self.get_side(
            mask, 'left', inp, inp_length=self.side_length).transpose(1, 2)
        right_input = self.get_side(
            mask, 'right', inp, inp_length=self.side_length).transpose(1, 2)

        for l_index in range(max_mask_len // 2):
            left_output = self.forward_step_left(left_input)
            outputs[:, l_index] = left_output
            left_input = torch.cat(
                (left_input, left_output.unsqueeze(2)), 2)[:, :, 1:]

            right_output = self.forward_step_right(right_input)
            outputs[:, max_mask_len - l_index - 1] = right_output
            right_input = torch.cat(
                (right_output.unsqueeze(2), right_input), 2)[:, :, :-1]

        return outputs

    def forward_step_left(self, left_inputs):
        out1 = self.left_conv1(left_inputs)
        out2 = self.left_conv2(left_inputs)
        out3 = self.left_conv3(left_inputs)
        out4 = self.left_conv4(left_inputs)
        out5 = self.left_conv5(left_inputs)
        out6 = self.left_conv6(left_inputs)
        out_summed = out1 + out2 + out3 + out4 + out5 + out6

        down_out = self.left_downscale_time_conv(out_summed)
        mean_out = torch.mean(down_out, 2)

        linear_out = self.linear_left(mean_out)

        if self.verbose:
            print('\nLEFT INPUTS')
            print('\nleft_inputs\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
                torch.mean(left_inputs), torch.std(left_inputs), torch.min(left_inputs), torch.max(left_inputs), left_inputs.size()))

            print('\nout1\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
                torch.mean(out1), torch.std(out1), torch.min(out1), torch.max(out1), out1.size()))

            print('\nout2\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
                torch.mean(out2), torch.std(out2), torch.min(out2), torch.max(out2), out2.size()))

            print('\nout3\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
                torch.mean(out3), torch.std(out3), torch.min(out3), torch.max(out3), out3.size()))

            print('\nout4\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
                torch.mean(out4), torch.std(out4), torch.min(out4), torch.max(out4), out4.size()))

            print('\nout5\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
                torch.mean(out5), torch.std(out5), torch.min(out5), torch.max(out5), out5.size()))

            print('\nout6\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
                torch.mean(out6), torch.std(out6), torch.min(out6), torch.max(out6), out6.size()))

            print('\n out_summed\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
                torch.mean(out_summed), torch.std(out_summed), torch.min(out_summed), torch.max(out_summed), out_summed.size()))

            print('\n down_out\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
                torch.mean(down_out), torch.std(down_out), torch.min(down_out), torch.max(down_out), down_out.size()))

            print('\n mean_out\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
                torch.mean(mean_out), torch.std(mean_out), torch.min(mean_out), torch.max(mean_out), mean_out.size()))

            print('\n linear_out\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
                torch.mean(linear_out), torch.std(linear_out), torch.min(linear_out), torch.max(linear_out), linear_out.size()))

        return linear_out

    def forward_step_right(self, right_inputs):
        out1 = self.right_conv1(right_inputs)
        out2 = self.right_conv2(right_inputs)
        out3 = self.right_conv3(right_inputs)
        out4 = self.right_conv4(right_inputs)
        out5 = self.right_conv5(right_inputs)
        out6 = self.right_conv6(right_inputs)
        out_summed = out1 + out2 + out3 + out4 + out5 + out6

        down_out = self.right_downscale_time_conv(out_summed)
        mean_out = torch.mean(down_out, 2)

        linear_out = self.linear_right(mean_out)

        if self.verbose:
            print('\nRIGHT INPUTS')
            print('\nright_inputs\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
                torch.mean(right_inputs), torch.std(right_inputs), torch.min(right_inputs), torch.max(right_inputs), right_inputs.size()))

            print('\nout1\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
                torch.mean(out1), torch.std(out1), torch.min(out1), torch.max(out1), out1.size()))

            print('\nout2\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
                torch.mean(out2), torch.std(out2), torch.min(out2), torch.max(out2), out2.size()))

            print('\nout3\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
                torch.mean(out3), torch.std(out3), torch.min(out3), torch.max(out3), out3.size()))

            print('\nout4\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
                torch.mean(out4), torch.std(out4), torch.min(out4), torch.max(out4), out4.size()))

            print('\nout5\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
                torch.mean(out5), torch.std(out5), torch.min(out5), torch.max(out5), out5.size()))

            print('\nout6\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
                torch.mean(out6), torch.std(out6), torch.min(out6), torch.max(out6), out6.size()))

            print('\n out_summed\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
                torch.mean(out_summed), torch.std(out_summed), torch.min(out_summed), torch.max(out_summed), out_summed.size()))

            print('\n down_out\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
                torch.mean(down_out), torch.std(down_out), torch.min(down_out), torch.max(down_out), down_out.size()))

            print('\n mean_out\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
                torch.mean(mean_out), torch.std(mean_out), torch.min(mean_out), torch.max(mean_out), mean_out.size()))

            print('\n linear_out\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
                torch.mean(linear_out), torch.std(linear_out), torch.min(linear_out), torch.max(linear_out), linear_out.size()))

        return linear_out

    def get_side(self, mask, side, inputs, inp_length):
        with torch.no_grad():
            side_inputs = []

            for batch_index, mask_batch in enumerate(mask):
                mask_len = torch.sum(mask_batch).int()

                if mask_len == 0:
                    side_input = torch.zeros(
                        (inp_length, inputs.size(2))).to(mask.device)
                else:

                    m_nonzero = mask_batch.nonzero().flatten()
                    first_nonzero = m_nonzero[0]
                    last_nonzero = m_nonzero[-1]

                    if side == 'left':
                        end_index = first_nonzero - 1
                        start_index = max(0, end_index - inp_length)
                    elif side == 'right':
                        start_index = last_nonzero + 1
                        end_index = min(inputs[batch_index].size(
                            1), start_index + inp_length)

                    side_input = inputs[batch_index][start_index:end_index]

                    if end_index - start_index < inp_length:
                        pad_zeros = torch.zeros(
                            (inp_length - side_input.shape[0], side_input.shape[1])).to(mask.device)
                        if side == 'left':
                            side_input = torch.cat((pad_zeros, side_input), 0)
                        elif side == 'right':
                            side_input = torch.cat((side_input, pad_zeros), 0)

                side_inputs.append(side_input)

        return torch.stack(side_inputs)

    def model_summary(self, model):
        print("model_summary")
        print()
        print("Layer_name" + "\t" * 7 + "Number of Parameters")
        print("=" * 100)
        model_parameters = [
            layer for layer in model.parameters() if layer.requires_grad]
        layer_name = [child for child in model.children()]
        j = 0
        total_params = 0
        print("\t" * 10)
        for i in layer_name:
            print()
            param = 0
            try:
                bias = (i.bias is not None)
            except:
                bias = False
            if not bias:
                param = model_parameters[j].numel(
                ) + model_parameters[j + 1].numel()
                j = j + 2
            else:
                param = model_parameters[j].numel()
                j = j + 1
            print(str(i) + "\t" * 3 + str(param))
            total_params += param
        print("=" * 100)
        print(f"Total Params:{total_params}")
