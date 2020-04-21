import math
import torch
import numpy as np
from taylor_activation import TaylorActivation


class ReconstructionModel(torch.nn.Module):
    def __init__(self, kernel_size=25, kernel_size_step=-4, dropout=0.01, side_length=48, verbose=False):
        super(ReconstructionModel, self).__init__()
        self.verbose = verbose
        self.side_length = side_length

        self.dropout = torch.nn.Dropout(p=dropout)
        kernel_sizes = [kernel_size, kernel_size + kernel_size_step, kernel_size + 2 *
                        kernel_size_step, kernel_size + 3 * kernel_size_step, kernel_size + 4 * kernel_size_step, kernel_size + 5 * kernel_size_step]

        self.SIDES = ['left', 'right']
        self.FEATURE_DIM = 161

        self.linear = torch.nn.ModuleDict({})
        self.final_layer = torch.nn.Sequential(
            torch.nn.Linear(self.FEATURE_DIM * 2, int(self.FEATURE_DIM * 1.5)),
            TaylorActivation(),
            torch.nn.Linear(int(self.FEATURE_DIM * 1.5), self.FEATURE_DIM),
            TaylorActivation(clip_min=0, clip_max=6)
        )

        self.downscale_time_conv = torch.nn.ModuleDict({})
        self.autoencoder1_layer = torch.nn.ModuleDict({})
        self.autoencoder2_layer = torch.nn.ModuleDict({})

        for i, kernel_size in enumerate(kernel_sizes):
            self.autoencoder1_layer[f"{i}"] = self.conv_layer(
                in_channels=self.FEATURE_DIM, kernel_size=kernel_size)
            self.autoencoder2_layer[f"{i}"] = self.conv_layer(
                in_channels=self.FEATURE_DIM, kernel_size=kernel_size)

        for side in self.SIDES:
            self.linear[side] = torch.nn.Sequential(
                torch.nn.Linear(self.FEATURE_DIM // 8,
                                (self.FEATURE_DIM // 8 + self.FEATURE_DIM) // 2),
                TaylorActivation(),
                torch.nn.Linear(
                    (self.FEATURE_DIM // 8 + self.FEATURE_DIM) // 2, self.FEATURE_DIM),
                TaylorActivation()
            )

            self.downscale_time_conv[side] = torch.nn.Sequential(
                torch.nn.Conv1d(
                    in_channels=self.FEATURE_DIM // 8,
                    out_channels=self.FEATURE_DIM // 8,
                    kernel_size=2,
                    stride=1,
                    dilation=1,
                    padding=0
                ),
                TaylorActivation(),
                torch.nn.Conv1d(
                    in_channels=self.FEATURE_DIM // 8,
                    out_channels=self.FEATURE_DIM // 8,
                    kernel_size=2,
                    stride=1,
                    dilation=2,
                    padding=0
                ),
                TaylorActivation(),
                torch.nn.Conv1d(
                    in_channels=self.FEATURE_DIM // 8,
                    out_channels=self.FEATURE_DIM // 8,
                    kernel_size=2,
                    stride=1,
                    dilation=4,
                    padding=0
                ),
                TaylorActivation(),
                torch.nn.Conv1d(
                    in_channels=self.FEATURE_DIM // 8,
                    out_channels=self.FEATURE_DIM // 8,
                    kernel_size=2,
                    stride=1,
                    dilation=8,
                    padding=0
                ),
                TaylorActivation(),
                torch.nn.Conv1d(
                    in_channels=self.FEATURE_DIM // 8,
                    out_channels=self.FEATURE_DIM // 8,
                    kernel_size=2,
                    stride=1,
                    dilation=16,
                    padding=0,
                ),
                TaylorActivation(),
                self.dropout
            )

    def autoencoder1(self, inputs):
        out_summed = torch.zeros((inputs.size(0), inputs.size(
            1) // 8, inputs.size(2))).to(inputs.device)

        for key in self.autoencoder1_layer:
            out = self.autoencoder1_layer[key](inputs)
            if self.verbose:
                print('\n out1 ({}) \tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
                    key, torch.mean(out), torch.std(out), torch.min(out), torch.max(out), out.size()))

            out_summed += out

        return out_summed

    def autoencoder2(self, inputs):
        out_summed = torch.zeros(inputs.size()).to(inputs.device)

        for key in self.autoencoder2_layer:
            out = self.autoencoder2_layer[key](inputs)
            if self.verbose:
                print('\n out2 ({}) \tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
                    key, torch.mean(out), torch.std(out), torch.min(out), torch.max(out), out.size()))

            out_summed += out

        return out_summed

    def conv_layer(self, in_channels, kernel_size):
        return torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
            TaylorActivation(),
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
            self.dropout
        )

    def forward(self, x, mask):
        # if self.make_4d:
        #     x = x.view(x.size(0), x.size(3), x.size(2))

        inp = x

        mask_sums = torch.sum(mask, 1)

        # TODO: Consider using the mean mask length
        max_mask_len = torch.max(mask_sums).int().item()
        # print(f"max_mask_len: {max_mask_len}\tmask_sums: {mask_sums.size()}")
        # print(mask)

        if max_mask_len == 0:
            # Super weird edge case where the entire batch has no mask.
            # In this case, return a tensor of SEQ_LEN=1
            max_mask_len = 1

        outputs = torch.empty((inp.size(0), max_mask_len, inp.size(2))).to(x.device).fill_(-1)
        # print(f"\noutputs: {outputs.size()}\tmax_mask_len: {max_mask_len}")

        left_input, left_start_indices, left_end_indices = self.get_side(
            mask, 'left', inp)
        right_input, right_start_indices, right_end_indices = self.get_side(
            mask, 'right', inp)

        for l_index in range(math.ceil(max_mask_len / 2)):
            left_remain_input = torch.zeros(inp.size(0), self.side_length, inp.size(2)).to(mask.device)
            right_remain_input = torch.zeros(inp.size(0), self.side_length, inp.size(2)).to(mask.device)

            # First, we need to identify which left and right element we are predicting
            # We are adding / subtracting 2 because the self.get_side returns the start / end
            # index for the part before / after the mask.

            # Next, we need to get the remain inputs. This is the portion that is to the
            # right / left of the predict index
            for batch_index in range(len(mask)):
                # + 2 because the left_start_indices is where the mask ends. Then +1 is what we want to predict, so +2 is after that
                left_start = left_end_indices[batch_index] + 2 + l_index
                left_end = left_start + self.side_length
                left_end = min(left_end, inp[batch_index].size(0) - 1)

                right_start = left_start_indices[batch_index] - (2 + l_index)
                right_start = max(right_start, 0)
                right_end = right_start + self.side_length

                # print(f"left_start: {left_start}\tleft_end: {left_end}\tright_start: {right_start}\tright_end: {right_end}\tinp[batch_index]: {inp[batch_index].size()}")

                left_remain_input[batch_index][0:left_end - left_start] = inp[batch_index][left_start:left_end]
                right_remain_input[batch_index][0:right_end - right_start] = inp[batch_index][right_start:right_end]

            # Do forward passes for left and right
            left_output = self.forward_step(left_input, 'left')
            left_remain_output = self.forward_step(left_remain_input, 'right')
            left_combined_output = torch.cat(
                (left_output, left_remain_output), dim=1)
            left_final = self.final_layer(left_combined_output)

            right_output = self.forward_step(right_input, 'right')
            right_remain_output = self.forward_step(right_remain_input, 'left')
            right_combined_output = torch.cat(
                (right_output, right_remain_output), dim=1)
            right_final = self.final_layer(right_combined_output)

            # print(torch.mean(right_input, 2), torch.mean(right_input, 2).size())
            # Then, we need to update the output
            for batch_index in range(len(mask)):
                # We are doing this in a loop because each item of the batch has a different
                # size
                nonzeros = mask[batch_index].nonzero().flatten()
                if nonzeros.size(0) == 0:
                    # In the event that our mask didn't find anything, skip this item
                    continue

                # print(batch_index, nonzeros, outputs[batch_index].size(0), l_index, outputs.size())
                # In the case that there are multiple masked values, we may have to trim
                right_index = min(nonzeros[-1] - nonzeros[0] - l_index, outputs[batch_index].size(0) - l_index - 1)

                # if batch_index == 3:
                #     print(f"outputs[{batch_index}]: {outputs[batch_index].size()}\tl_index: {l_index}\tright_index: {right_index}")

                if l_index <= right_index:

                    outputs[batch_index, l_index, :] = left_final[batch_index]
                    outputs[batch_index, right_index, :] = right_final[batch_index]

                    # Finally, we need to update each side inputs with the new outputs
                    left_input[batch_index] = torch.cat((left_input[batch_index], left_final[batch_index].unsqueeze(0)), 0)[1:, :]
                    right_input[batch_index] = torch.cat((right_final[batch_index].unsqueeze(0), right_input[batch_index]), 0)[:-1, :]

            # print(torch.mean(right_input, 2))
            # print(torch.mean(left_final, 1))
            # quit()
        # print(torch.mean(outputs[0], dim=1))
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
        # print(inp.size())
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

    def get_side(self, mask, side, inputs, offset=0):
        """Returns all input elements from a particular `side`, given a `mask`

        Arguments:
            mask {[Tensor]} -- Tensor of mask to apply for reconstruction. Size = [BATCH x SEQ_LEN]
            side {string} -- 'left' or 'right'
            inputs {[Tensor]} -- The input spectrogram. Size = [BATCH x SEQ_LEN x CHANNELS]

        Keyword Arguments:
            offset {int} -- [description] (default: {0})

        Returns:
            ([Tensor], int, int) -- Tuple with Output spectrogram to the side of the input. Size = [BATCH x `self.side_length` x CHANNELS], start index and end_index
        """
        with torch.no_grad():
            side_inputs = torch.zeros(
                (mask.size(0), self.side_length, inputs.size(2))).to(mask.device)
            start_indices = torch.zeros(mask.size(0)).to(
                dtype=torch.int, device=mask.device)
            end_indices = torch.zeros(mask.size(0)).to(
                dtype=torch.int, device=mask.device)

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
                        start_index = max(0, end_index - self.side_length)
                    elif side == 'right':
                        start_index = last_nonzero + 1
                        end_index = min(inputs[batch_index].size(
                            0), start_index + self.side_length)

                    start_indices[batch_index] = start_index
                    end_indices[batch_index] = end_index

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
                                    : side_input.size(0), :] = side_input
        return side_inputs, start_indices, end_indices

    def fit_to_size(self, pred, sizes):
        """Shrinks or expands the prediction to the `size`

        Arguments:
            pred {Tensor} -- Tensor of predictions of shape: [BATCH x SEQ_LEN x CHANNELS]
            sizes {list of str} -- List of sizes. Shape: [SEQ_LEN]

        Returns:
            {Tensor} -- Fit tensor to sequence length of size of shape: [BATCH x SEQ_LEN x CHANNELS]
        """

        max_length = max(sizes)
        # print(f"\n\nmax_length: {max_length}")

        resized_pred = torch.zeros((pred.size(0), max_length, pred.size(2))).to(pred.device)
        for batch_index in range(len(sizes)):
            size = sizes[batch_index]
            first_nonzero_index = (torch.mean(pred[batch_index], dim=1) == -1).nonzero()
            if first_nonzero_index.size(0) == 0:
                # No need to rescale
                pred_part = pred[batch_index]
            else:
                first_nonzero_index = first_nonzero_index.flatten()[0]

                if first_nonzero_index == 0:
                    # If the first element is -1, it means there was no prediction, maybe due to no masking?
                    continue

                pred_part = pred[batch_index][0:first_nonzero_index]
            # print(f"\tBatch: {batch_index}\tpred_part: {pred_part.size()}\tsizes[batch_index]: {sizes[batch_index]}\tfirst_nonzero_index: {first_nonzero_index}")
            if pred_part.size(0) != sizes[batch_index]:
                # If our prediction is larger or smaller than the requested size, we need to interpolate
                pred_t = pred_part.unsqueeze(0).permute(0, 2, 1)  # Convert to BATCH x CHANNELS x SEQ_LEN

                # print(f"pred_t: {pred_t.size()}\t: size: {size}")
                pred_part = torch.nn.functional.interpolate(pred_t, size=size).permute(0, 2, 1).squeeze(0)
                # print(f"interpolated_pred: {interpolated_pred.size()}")
            # print(f"resized_pred[batch_index]: {resized_pred[batch_index].size()}")

            resized_pred[batch_index, 0:size] = pred_part

        # print(torch.mean(resized_pred, dim=2))

        return resized_pred

    def get_param_size(self):
        """[summary]

        Returns:
            [type] -- [description]
        """
        params = 0
        for p in self.parameters():
            tmp = 1
            for x in p.size():
                tmp *= x
            params += tmp
        return params
