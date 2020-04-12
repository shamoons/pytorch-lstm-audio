import torch


class ReconstructionModel(torch.nn.Module):
    def __init__(self, feature_dim=161, kernel_size=25, kernel_size_step=-4, make_4d=False, dropout=0.01, side_length=96, verbose=False):
        super(ReconstructionModel, self).__init__()
        self.make_4d = make_4d
        self.verbose = verbose
        self.side_length = side_length

        self.dropout = torch.nn.Dropout(p=dropout)
        kernel_sizes = [kernel_size, kernel_size + kernel_size_step, kernel_size + 2 *
                        kernel_size_step, kernel_size + 3 * kernel_size_step, kernel_size + 4 * kernel_size_step, kernel_size + 5 * kernel_size_step]

        self.SIDES = ['left', 'right']

        self.linear = torch.nn.ModuleDict({})
        self.conv1 = torch.nn.ModuleDict({})
        self.conv2 = torch.nn.ModuleDict({})
        self.conv3 = torch.nn.ModuleDict({})
        self.conv4 = torch.nn.ModuleDict({})
        self.conv5 = torch.nn.ModuleDict({})
        self.conv6 = torch.nn.ModuleDict({})
        self.downscale_time_conv = torch.nn.ModuleDict({})

        for side in self.SIDES:
            self.linear[side] = torch.nn.Sequential(
                torch.nn.Linear(feature_dim, feature_dim),
                torch.nn.PReLU(),
                torch.nn.Linear(feature_dim, feature_dim),
                torch.nn.ReLU6()

                # torch.nn.Linear(40, (feature_dim + 40) // 2),
                # torch.nn.Tanh(),
                # torch.nn.Linear((feature_dim + 40) // 2, feature_dim),
                # torch.nn.ReLU6()
            )

            # self.encoder_conv[side] = self.encoder(feature_dim, kernel_size, kernel_size_step)

            self.conv1[side] = self.conv_layer(in_channels=feature_dim,
                                               kernel_size=kernel_sizes[0])
            self.conv2[side] = self.conv_layer(in_channels=feature_dim,
                                               kernel_size=kernel_sizes[1])
            self.conv3[side] = self.conv_layer(in_channels=feature_dim,
                                               kernel_size=kernel_sizes[2])
            self.conv4[side] = self.conv_layer(in_channels=feature_dim,
                                               kernel_size=kernel_sizes[3])
            self.conv5[side] = self.conv_layer(in_channels=feature_dim,
                                               kernel_size=kernel_sizes[4])
            self.conv6[side] = self.conv_layer(in_channels=feature_dim,
                                               kernel_size=kernel_sizes[5])

            self.downscale_time_conv[side] = torch.nn.Sequential(
                torch.nn.Conv1d(
                    in_channels=feature_dim ,
                    out_channels=feature_dim ,
                    kernel_size=2,
                    stride=1,
                    dilation=1,
                    padding=0
                ),
                torch.nn.Tanh(),
                torch.nn.Conv1d(
                    in_channels=feature_dim ,
                    out_channels=feature_dim ,
                    kernel_size=2,
                    stride=1,
                    dilation=2,
                    padding=0
                ),
                torch.nn.Tanh(),
                torch.nn.Conv1d(
                    in_channels=feature_dim ,
                    out_channels=feature_dim ,
                    kernel_size=2,
                    stride=1,
                    dilation=4,
                    padding=0
                ),
                torch.nn.Tanh(),
                torch.nn.Conv1d(
                    in_channels=feature_dim ,
                    out_channels=feature_dim ,
                    kernel_size=2,
                    stride=1,
                    dilation=8,
                    padding=0
                ),
                torch.nn.Tanh(),
                torch.nn.Conv1d(
                    in_channels=feature_dim ,
                    out_channels=feature_dim ,
                    kernel_size=2,
                    stride=1,
                    dilation=16,
                    padding=0,
                    # groups=feature_dim // 4
                ),
                torch.nn.Tanh()
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
            # torch.nn.PReLU(),
            torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
            torch.nn.PReLU(),
            torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
            torch.nn.PReLU(),
            torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
            torch.nn.PReLU(),
            # torch.nn.Conv1d(
            #     in_channels=in_channels // 4,
            #     out_channels=in_channels // 4,
            #     kernel_size=kernel_size,
            #     stride=1,
            #     padding=kernel_size // 2,
            #     groups=in_channels // 4
            # ),
            # torch.nn.PReLU()
        )

    def forward(self, x, mask):
        if self.make_4d:
            x = x.view(x.size(0), x.size(3), x.size(2))

        inp = x

        mask_sums = torch.sum(mask, 1)

        # TODO: Consider using the mean mask length
        max_mask_len = torch.max(mask_sums).int()

        outputs = torch.zeros((inp.size(0), max_mask_len, inp.size(2))).to(x.device)
        print(f"inp: {inp.size()}\t mask: {mask.size()}")
        left_input = self.get_side(
            mask, 'left', inp, inp_length=self.side_length)
        # right_input = self.get_side(
        #     mask, 'right', inp, inp_length=self.side_length)

        for l_index in range(max_mask_len // 2):
            left_output = self.forward_step(left_input, 'left')

            # Need to get the input including zeros
            # right_left_input = self.get_side(mask, 'right', inp, inp_length=self.side_length, offset=l_index + 1)

            print(f"inp: {torch.mean(inp[0], 1)}\t{inp.size()}")
            print(f"left_input: {torch.mean(left_input[0], 1)}\t{left_input.size()}")
            # print(f"right_left_input: {torch.mean(right_left_input[0], 1)}\t{right_left_input.size()}")
            quit()


            right_left_output = self.forward_step(right_left_input, 'right')

            left_output = torch.max(left_output, right_left_output)
            
            outputs[:, l_index] = left_output

            right_output = self.forward_step(right_input, 'right')
            # Need to get the input including zeros
            left_right_input = self.get_side(
                mask, 'left', inp, inp_length=self.side_length, offset=l_index + 1)
            left_right_output = self.forward_step(left_right_input, 'left')
            right_output = torch.max(right_output, left_right_output)

            outputs[:, max_mask_len - l_index - 1] = right_output

            print(torch.mean(outputs[0], 1))
            quit()

            left_input = torch.cat(
                (left_input, left_output.unsqueeze(2)), 2)[:, :, 1:]

            right_input = torch.cat(
                (right_output.unsqueeze(2), right_input), 2)[:, :, :-1]

            for batch_index in range(inp.size(0)):
                batch_mask = mask[batch_index]
                batch_inp = inp[batch_index]
                batch_out = outputs[batch_index]
                batch_mask_sum = torch.sum(batch_mask).int()

                if batch_mask_sum == 0:
                    continue

                # Converts to [1 x CHANNELS x SEQ]
                batch_out_t = batch_out.transpose(0, 1).unsqueeze(0)

                # print(f"batch_mask: {batch_mask.size()}\tbatch_inp: {batch_inp.size()}\tbatch_out: {batch_out.size()}\tbatch_mask_sum: {batch_mask_sum}\tbatch_out_t: {batch_out_t.size()}")

                # Interpolate down to the largest mask that we have in this batch
                batch_out_t = torch.nn.functional.interpolate(batch_out_t, size=batch_mask_sum.item())

                # Convert back to [SEQ x CHANNELS]
                batch_out = torch.squeeze(batch_out_t).transpose(0, 1)
                inp[batch_index][batch_mask == 1] = batch_out

        # quit()
            

        return outputs

    def forward_step(self, inp, side):
        """Forward steps for a particular input
        
        Arguments:
            inp {[Tensor]} -- Size: [BATCH x SEQ_LEN x CHANNELS]
            side {[string]} -- Which side to forward: `left` / `right`
        
        Returns:
            [Tensor] -- [description]
        """
        inputs = inp.clone().permute(0, 2, 1)

        out1 = self.conv1[side](inputs)
        out2 = self.conv2[side](inputs)
        out3 = self.conv3[side](inputs)
        out4 = self.conv4[side](inputs)
        out5 = self.conv5[side](inputs)
        out6 = self.conv6[side](inputs)
        # out_summed = out1 + out2 + out3 + out4 + out5 + out6
        out_summed = torch.max(out1, out2)
        out_summed = torch.max(out_summed, out3)
        out_summed = torch.max(out_summed, out4)
        out_summed = torch.max(out_summed, out5)
        out_summed = torch.max(out_summed, out6)

        down_out = self.downscale_time_conv[side](out_summed)
        mean_out = torch.mean(down_out, 2)

        linear_out = self.linear[side](mean_out)

        if self.verbose and side == 'left':
            print(f"\nSIDE: {side}")
            print('\ninputs\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
                torch.mean(inputs), torch.std(inputs), torch.min(inputs), torch.max(inputs), inputs.size()))

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

    def get_side(self, mask, side, inputs, inp_length, offset=0):
        """Returns all input elements from a particular `side`, given a `mask`
        
        Arguments:
            mask {[Tensor]} -- Tensor of mask to apply for reconstruction. Size = [BATCH x SEQ_LEN]
            side {[string]} -- 'left' or 'right'
            inputs {[Tensor]} -- The input spectrogram. Size = [BATCH x SEQ_LEN x CHANNELS]
            inp_length {[int]} -- The length of the input to return
        
        Keyword Arguments:
            offset {int} -- [description] (default: {0})
        
        Returns:
            [Tensor] -- Output spectrogram to the side of the input. Size = [BATCH x `inp_length` x CHANNELS]
        """
        with torch.no_grad():
            side_inputs = torch.zeros(
                (mask.size(0), inp_length, inputs.size(2))).to(mask.device)

            # print(f"inputs: {inputs.size()}\tside_inputs: {side_inputs.size()}\tmask: {mask.size()}")

            for batch_index, mask_batch in enumerate(mask):
                mask_len = torch.sum(mask_batch).int()

                if mask_len == 0:
                    continue
                else:
                    m_nonzero = mask_batch.nonzero().flatten()
                    first_nonzero = m_nonzero[0] + offset
                    last_nonzero = m_nonzero[-1] - offset

                    print(f"first_nonzero: {first_nonzero}\t{inputs[batch_index, first_nonzero].mean()}")


                    if side == 'left':
                        end_index = first_nonzero - 1
                        start_index = max(0, end_index - inp_length)
                    elif side == 'right':
                        start_index = last_nonzero + 1
                        end_index = min(inputs[batch_index].size(1), start_index + inp_length)
                    
                    side_input = inputs[batch_index,start_index:end_index, :]
                    # print(f"side_inputs: {side_inputs.size()}")
                    # print(f"side_input: {torch.mean(side_input, 0)}\t{side_input.size()}")
                    # quit()

                    # print(f"side: {side}\tside_inputs: {side_inputs.size()}\tside_input: {side_input.size()}\tleft: {side_inputs[batch_index,-side_input.size(0):,:].size()}\tright: {side_inputs[batch_index,:side_input.size(0),:].size()}")

                    if side == 'left':
                        side_inputs[batch_index, -side_input.size(0):, :] = side_input
                    elif side == 'right':
                        side_inputs[batch_index,:side_input.size(0), :] = side_input
        return side_inputs

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
