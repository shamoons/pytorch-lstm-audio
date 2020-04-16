import torch
from taylor_activation import TaylorActivation


class ReconstructionModel(torch.nn.Module):
    def __init__(self, feature_dim=161, kernel_size=25, kernel_size_step=-4, make_4d=False, dropout=0.01, side_length=48, verbose=False):
        super(ReconstructionModel, self).__init__()
        self.make_4d = make_4d
        self.verbose = verbose
        self.side_length = side_length

        self.dropout = torch.nn.Dropout(p=dropout)
        kernel_sizes = [kernel_size, kernel_size + kernel_size_step, kernel_size + 2 *
                        kernel_size_step, kernel_size + 3 * kernel_size_step, kernel_size + 4 * kernel_size_step, kernel_size + 5 * kernel_size_step]

        self.SIDES = ['left', 'right']

        self.linear = torch.nn.ModuleDict({})
        self.final_layer = torch.nn.Sequential(
            torch.nn.Linear(feature_dim * 2, int(feature_dim * 1.5)),
            TaylorActivation(),
            torch.nn.Linear(int(feature_dim * 1.5), feature_dim),
            TaylorActivation(clip_min=0, clip_max=6)
        )

        self.downscale_time_conv = torch.nn.ModuleDict({})
        self.autoencoder1_layer = torch.nn.ModuleDict({})
        self.autoencoder2_layer = torch.nn.ModuleDict({})

        for i, kernel_size in enumerate(kernel_sizes):
            self.autoencoder1_layer[f"{i}"] = self.conv_layer(in_channels=feature_dim, kernel_size=kernel_size)
            self.autoencoder2_layer[f"{i}"] = self.conv_layer(in_channels=feature_dim, kernel_size=kernel_size)

        for side in self.SIDES:
            self.linear[side] = torch.nn.Sequential(
                torch.nn.Linear(feature_dim // 16, (feature_dim // 16 + feature_dim) // 2),
                TaylorActivation(),
                torch.nn.Linear((feature_dim // 16 + feature_dim) // 2, feature_dim),
                TaylorActivation()
            )

            self.downscale_time_conv[side] = torch.nn.Sequential(
                torch.nn.Conv1d(
                    in_channels=feature_dim // 16,
                    out_channels=feature_dim // 16,
                    kernel_size=2,
                    stride=1,
                    dilation=1,
                    padding=0
                ),
                TaylorActivation(),
                torch.nn.Conv1d(
                    in_channels=feature_dim // 16,
                    out_channels=feature_dim // 16,
                    kernel_size=2,
                    stride=1,
                    dilation=2,
                    padding=0
                ),
                TaylorActivation(),
                torch.nn.Conv1d(
                    in_channels=feature_dim // 16,
                    out_channels=feature_dim // 16,
                    kernel_size=2,
                    stride=1,
                    dilation=4,
                    padding=0
                ),
                TaylorActivation(),
                torch.nn.Conv1d(
                    in_channels=feature_dim // 16,
                    out_channels=feature_dim // 16,
                    kernel_size=2,
                    stride=1,
                    dilation=8,
                    padding=0
                ),
                TaylorActivation(),
                torch.nn.Conv1d(
                    in_channels=feature_dim // 16,
                    out_channels=feature_dim // 16,
                    kernel_size=2,
                    stride=1,
                    dilation=16,
                    padding=0,
                ),
                TaylorActivation(),
                self.dropout
            )

    def autoencoder1(self, inputs):
        out_summed = torch.zeros((inputs.size(0), inputs.size(1) // 16, inputs.size(2))).to(inputs.device)

        for key in self.autoencoder1_layer:
            out = self.autoencoder1_layer[key](inputs)
            if self.verbose:
                print('\n out1 ({}) \tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(key, torch.mean(out), torch.std(out), torch.min(out), torch.max(out), out.size()))

            out_summed += out
    
        return out_summed

    def autoencoder2(self, inputs):
        out_summed = torch.zeros(inputs.size()).to(inputs.device)

        for key in self.autoencoder2_layer:
            out = self.autoencoder2_layer[key](inputs)
            if self.verbose:
                print('\n out2 ({}) \tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(key, torch.mean(out), torch.std(out), torch.min(out), torch.max(out), out.size()))

            out_summed += out
    
        return out_summed

    def conv_layer(self, in_channels, kernel_size):
        return torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
            TaylorActivation(),
            torch.nn.Conv1d(
                in_channels=in_channels // 2,
                out_channels=in_channels // 4,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
            TaylorActivation(),
            torch.nn.Conv1d(
                in_channels=in_channels // 4,
                out_channels=in_channels // 8,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
            TaylorActivation(),
            torch.nn.Conv1d(
                in_channels=in_channels // 8,
                out_channels=in_channels // 16,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
            TaylorActivation(),
            self.dropout
        )

    def forward(self, x, mask):
        if self.make_4d:
            x = x.view(x.size(0), x.size(3), x.size(2))

        inp = x

        mask_sums = torch.sum(mask, 1)

        # TODO: Consider using the mean mask length
        max_mask_len = torch.max(mask_sums).int().item()
        # print(f"max_mask_len: {max_mask_len}")

        if max_mask_len == 0:
            # Super weird edge case where the entire batch has no mask.
            # In this case, return a tensor of SEQ_LEN=1
            max_mask_len = 1

        outputs = torch.zeros((inp.size(0), max_mask_len, inp.size(2)), requires_grad=True).to(x.device)

        left_input = self.get_side(
            mask, 'left', inp, inp_length=self.side_length)
        right_input = self.get_side(
            mask, 'right', inp, inp_length=self.side_length)

        for l_index in range(max_mask_len // 2):
            left_output = self.forward_step(left_input, 'left')

            # Need to get the input including zeros
            right_left_input = self.get_side(
                mask, 'right', inp, inp_length=self.side_length, offset=max_mask_len - (l_index + 0))

            right_left_output = self.forward_step(right_left_input, 'right')

            combined_output = torch.cat((left_output, right_left_output), dim=1)
            outputs[:, l_index] = self.final_layer(combined_output)

            right_output = self.forward_step(right_input, 'right')
            # Need to get the input including zeros
            left_right_input = self.get_side(
                mask, 'left', inp, inp_length=self.side_length, offset=max_mask_len + (l_index + 1))
            left_right_output = self.forward_step(left_right_input, 'left')

            combined_output = torch.cat((right_output, left_right_output), dim=1)
            
            outputs[:, max_mask_len - l_index - 1] =  self.final_layer(combined_output)

            left_input = torch.cat(
                (left_input, left_output.unsqueeze(1)), 1)[:, 1:, :]

            right_input = torch.cat(
                (right_output.unsqueeze(1), right_input), 1)[:, :-1, :]

            for batch_index in range(inp.size(0)):
                batch_mask = mask[batch_index]
                batch_out = outputs[batch_index]
                batch_mask_sum = torch.sum(batch_mask).int()

                if batch_mask_sum == 0:
                    continue

                # Converts to [1 x CHANNELS x SEQ]
                batch_out_t = batch_out.permute(1, 0).unsqueeze(0)


                # Interpolate down to the largest mask that we have in this batch
                batch_out_t = torch.nn.functional.interpolate(
                    batch_out_t, size=batch_mask_sum.item())

                # Convert back to [SEQ x CHANNELS]
                batch_out = torch.squeeze(batch_out_t, dim=0).permute(1, 0)
                inp[batch_index][batch_mask == 1] = batch_out

        # print(f"outputs: {outputs.size()}")
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

        autoencoded_out1 = self.autoencoder1(inputs)
        # autoencoded_out2 = self.autoencoder2(autoencoded_out1)

        down_out = self.downscale_time_conv[side](autoencoded_out1)
        mean_out = torch.mean(down_out, 2)

        linear_out = self.linear[side](mean_out)

        if self.verbose and side == 'left':
            print(f"\nSIDE: {side}")
            print('\ninputs\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
                torch.mean(inputs), torch.std(inputs), torch.min(inputs), torch.max(inputs), inputs.size()))

            print('\n autoencoded_out1\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
                torch.mean(autoencoded_out1), torch.std(autoencoded_out1), torch.min(autoencoded_out1), torch.max(autoencoded_out1), autoencoded_out1.size()))

            # print('\n autoencoded_out2\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
            #     torch.mean(autoencoded_out2), torch.std(autoencoded_out2), torch.min(autoencoded_out2), torch.max(autoencoded_out2), autoencoded_out2.size()))

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
                mask_len = torch.sum(mask_batch).int().item()

                if mask_len == 0:
                    continue
                else:
                    m_nonzero = mask_batch.nonzero().flatten()
                    first_nonzero = m_nonzero[0] + offset
                    last_nonzero = m_nonzero[-1] - offset

                    if side == 'left':
                        end_index = first_nonzero - 1
                        start_index = max(0, end_index - inp_length)
                    elif side == 'right':
                        start_index = last_nonzero + 1
                        end_index = min(inputs[batch_index].size(
                            0), start_index + inp_length)

                    # print(f"\n\nfirst_nonzero: {first_nonzero}\tlast_nonzero: {last_nonzero}\tstart_index:{start_index}\tend_index: {end_index}\tindices: [{start_index}:{end_index}]\tmask_len: {mask_len}\toffset: {offset}")

                    if end_index - start_index <= 0:
                        continue
                    side_input = inputs[batch_index, start_index:end_index, :]

                    # print(f"\tbatch_inp: {inputs[batch_index].size()}\tside: {side}\tside_inputs: {side_inputs.size()}\tside_input: {side_input.size()}\tleft: {side_inputs[batch_index,-side_input.size(0):,:].size()}\tright: {side_inputs[batch_index,:side_input.size(0),:].size()}")

                    if side_input.size(0) == 0:
                        continue

                    if side == 'left':
                        side_inputs[batch_index, -
                                    side_input.size(0):, :] = side_input
                    elif side == 'right':
                        side_inputs[batch_index,
                                    :side_input.size(0), :] = side_input
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
