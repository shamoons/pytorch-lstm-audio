import torch
import numpy as np


class ReconstructionModel(torch.nn.Module):
    def __init__(self, feature_dim=161, kernel_size=25, kernel_size_step=-4, final_kernel_size=25, make_4d=False, dropout=0.01, verbose=False):
        super(ReconstructionModel, self).__init__()
        self.make_4d = make_4d
        self.verbose = verbose

        self.dropout = torch.nn.Dropout(p=dropout)

        kernel_sizes = [kernel_size, kernel_size + kernel_size_step, kernel_size + 2 *
                        kernel_size_step, kernel_size + 3 * kernel_size_step, kernel_size + 4 * kernel_size_step, kernel_size + 5 * kernel_size_step]

        self.left_conv1 = self.conv_layer(in_channels=feature_dim,
                        kernel_size=kernel_sizes[0])
        self.left_conv2 = self.conv_layer(in_channels=feature_dim,
                        kernel_size=kernel_sizes[1])
        self.left_conv3 = self.conv_layer(in_channels=feature_dim,
                        kernel_size=kernel_sizes[2])
        self.left_conv4 = self.conv_layer(in_channels=feature_dim,
                        kernel_size=kernel_sizes[3])
        self.left_conv5 = self.conv_layer(in_channels=feature_dim,
                        kernel_size=kernel_sizes[4])
        self.left_conv6 = self.conv_layer(in_channels=feature_dim,
                        kernel_size=kernel_sizes[5])



        self.right_conv1 = self.conv_layer(in_channels=feature_dim,
                        kernel_size=kernel_sizes[0])
        self.right_conv2 = self.conv_layer(in_channels=feature_dim,
                        kernel_size=kernel_sizes[1])
        self.right_conv3 = self.conv_layer(in_channels=feature_dim,
                        kernel_size=kernel_sizes[2])
        self.right_conv4 = self.conv_layer(in_channels=feature_dim,
                        kernel_size=kernel_sizes[3])
        self.right_conv5 = self.conv_layer(in_channels=feature_dim,
                        kernel_size=kernel_sizes[4])
        self.right_conv6 = self.conv_layer(in_channels=feature_dim,
                        kernel_size=kernel_sizes[5])

        self.upscale_conv = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=feature_dim // 16,
                out_channels=feature_dim // 8,
                kernel_size=final_kernel_size,
                stride=1,
                padding=final_kernel_size // 2
            ),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                in_channels=feature_dim // 8,
                out_channels=feature_dim // 4,
                kernel_size=final_kernel_size,
                stride=1,
                padding=final_kernel_size // 2
            ),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                in_channels=feature_dim // 4,
                out_channels=feature_dim // 2,
                kernel_size=final_kernel_size,
                stride=1,
                padding=final_kernel_size // 2
            ),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                in_channels=feature_dim // 2,
                out_channels=feature_dim,
                kernel_size=final_kernel_size,
                stride=1,
                padding=final_kernel_size // 2
            ),
            torch.nn.ReLU()
        )

        self.final_conv = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=feature_dim,
                out_channels=feature_dim,
                kernel_size=final_kernel_size,
                stride=1,
                padding=final_kernel_size // 2,
                groups=23
            ),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                in_channels=feature_dim,
                out_channels=feature_dim,
                kernel_size=final_kernel_size,
                stride=1,
                padding=final_kernel_size // 2,
                groups=7
            ),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                in_channels=feature_dim,
                out_channels=feature_dim,
                kernel_size=final_kernel_size,
                stride=1,
                padding=final_kernel_size // 2,
                groups=feature_dim
            ),
            torch.nn.ReLU6()
        )

    def forward(self, x, mask):
        if self.make_4d:
            x = x.view(x.size(0), x.size(3), x.size(2))

        left_inputs = self.get_side(mask, 'left', x).transpose(1, 2)
        right_inputs = self.get_side(mask, 'right', x).transpose(1, 2)

        if self.verbose:
            print(f"\nleft_inputs\tMean: {torch.mean(left_inputs):.4g} ± {torch.std(left_inputs):.4g}\tMin: {torch.min(left_inputs):.4g}\tMax: {torch.max(left_inputs):.4g}\tSize: {left_inputs.size()}")

            print(f"\nright_inputs\tMean: {torch.mean(right_inputs):.4g} ± {torch.std(right_inputs):.4g}\tMin: {torch.min(right_inputs):.4g}\tMax: {torch.max(right_inputs):.4g}\tSize: {right_inputs.size()}")

        left_out1 = self.left_conv1(left_inputs)
        right_out1 = self.right_conv1(right_inputs)

        left_out2 = self.left_conv2(left_inputs)
        right_out2 = self.right_conv2(right_inputs)

        left_out3 = self.left_conv3(left_inputs)
        right_out3 = self.right_conv3(right_inputs)

        left_out4 = self.left_conv4(left_inputs)
        right_out4 = self.right_conv4(right_inputs)

        left_out5 = self.left_conv5(left_inputs)
        right_out5 = self.right_conv5(right_inputs)

        left_out6 = self.left_conv6(left_inputs)
        right_out6 = self.right_conv6(right_inputs)

        left_summed = left_out1 + left_out2 + left_out3 + left_out4 + left_out5 + left_out6
        right_summed = right_out1 + right_out2 + right_out3 + right_out4 + right_out5 + right_out6

        out_summed = left_summed + right_summed

        upscale_out = self.upscale_conv(out_summed)

        upscale_out += left_inputs + right_inputs

        out = self.final_conv(upscale_out)

        if self.verbose:
            print(f"\n left_out1\tMean: {torch.mean(left_out1):.4g} ± {torch.std(left_out1):.4g}\tMin: {torch.min(left_out1):.4g}\tMax: {torch.max(left_out1):.4g}\tSize: {left_out1.size()}")

            print(f"\n right_out1\tMean: {torch.mean(right_out1):.4g} ± {torch.std(right_out1):.4g}\tMin: {torch.min(right_out1):.4g}\tMax: {torch.max(right_out1):.4g}\tSize: {right_out1.size()}")

            print(f"\n left_out2\tMean: {torch.mean(left_out2):.4g} ± {torch.std(left_out2):.4g}\tMin: {torch.min(left_out2):.4g}\tMax: {torch.max(left_out2):.4g}\tSize: {left_out2.size()}")

            print(f"\n right_out2\tMean: {torch.mean(right_out2):.4g} ± {torch.std(right_out2):.4g}\tMin: {torch.min(right_out2):.4g}\tMax: {torch.max(right_out2):.4g}\tSize: {right_out2.size()}")

            print(f"\n left_summed\tMean: {torch.mean(left_summed):.4g} ± {torch.std(left_summed):.4g}\tMin: {torch.min(left_summed):.4g}\tMax: {torch.max(left_summed):.4g}\tSize: {left_summed.size()}")

            print(f"\n right_summed\tMean: {torch.mean(right_summed):.4g} ± {torch.std(right_summed):.4g}\tMin: {torch.min(right_summed):.4g}\tMax: {torch.max(right_summed):.4g}\tSize: {right_summed.size()}")

            print(f"\n out_summed\tMean: {torch.mean(out_summed):.4g} ± {torch.std(out_summed):.4g}\tMin: {torch.min(out_summed):.4g}\tMax: {torch.max(out_summed):.4g}\tSize: {out_summed.size()}")

            print(f"\n upscale_out\tMean: {torch.mean(upscale_out):.4g} ± {torch.std(upscale_out):.4g}\tMin: {torch.min(upscale_out):.4g}\tMax: {torch.max(upscale_out):.4g}\tSize: {upscale_out.size()}")

            print(f"\n out\tMean: {torch.mean(out):.4g} ± {torch.std(out):.4g}\tMin: {torch.min(out):.4g}\tMax: {torch.max(out):.4g}\tSize: {out.size()}")

        out = out.transpose(1, 2)

        if self.make_4d:
            out = out.reshape(out.size(0), 1, out.size(2), out.size(1))

        return out

    # def __init__(self, feature_dim=161, kernel_size=25, kernel_size_step=-4, final_kernel_size=25, make_4d=False, dropout=0.01, verbose=False):
    #     super(ReconstructionModel, self).__init__()
    #     self.make_4d = make_4d
    #     self.verbose = verbose

    #     self.dropout = torch.nn.Dropout(p=dropout)

    #     kernel_sizes = [kernel_size, kernel_size + kernel_size_step, kernel_size + 2 *
    #                     kernel_size_step, kernel_size + 3 * kernel_size_step, kernel_size + 4 * kernel_size_step, kernel_size + 5 * kernel_size_step]
    #     self.conv1 = self.conv_layer(
    #         in_channels=feature_dim, kernel_size=kernel_sizes[0])
    #     self.conv2 = self.conv_layer(
    #         in_channels=feature_dim, kernel_size=kernel_sizes[1])
    #     self.conv3 = self.conv_layer(
    #         in_channels=feature_dim, kernel_size=kernel_sizes[2])
    #     self.conv4 = self.conv_layer(
    #         in_channels=feature_dim, kernel_size=kernel_sizes[3])
    #     self.conv5 = self.conv_layer(
    #         in_channels=feature_dim, kernel_size=kernel_sizes[4])
    #     self.conv6 = self.conv_layer(
    #         in_channels=feature_dim, kernel_size=kernel_sizes[5])

    #     self.upscale_conv = torch.nn.Sequential(
    #         torch.nn.Conv1d(
    #             in_channels=feature_dim // 16,
    #             out_channels=feature_dim // 8,
    #             kernel_size=final_kernel_size,
    #             stride=1,
    #             padding=final_kernel_size // 2
    #         ),
    #         torch.nn.ReLU(),
    #         torch.nn.Conv1d(
    #             in_channels=feature_dim // 8,
    #             out_channels=feature_dim // 4,
    #             kernel_size=final_kernel_size,
    #             stride=1,
    #             padding=final_kernel_size // 2
    #         ),
    #         torch.nn.ReLU(),
    #         torch.nn.Conv1d(
    #             in_channels=feature_dim // 4,
    #             out_channels=feature_dim // 2,
    #             kernel_size=final_kernel_size,
    #             stride=1,
    #             padding=final_kernel_size // 2
    #         ),
    #         torch.nn.ReLU(),
    #         torch.nn.Conv1d(
    #             in_channels=feature_dim // 2,
    #             out_channels=feature_dim,
    #             kernel_size=final_kernel_size,
    #             stride=1,
    #             padding=final_kernel_size // 2
    #         ),
    #         torch.nn.ReLU()
    #     )

    #     self.final_conv = torch.nn.Sequential(
    #         torch.nn.Conv1d(
    #             in_channels=feature_dim,
    #             out_channels=feature_dim,
    #             kernel_size=final_kernel_size,
    #             stride=1,
    #             padding=final_kernel_size // 2,
    #             groups=23
    #         ),
    #         torch.nn.ReLU(),
    #         torch.nn.Conv1d(
    #             in_channels=feature_dim,
    #             out_channels=feature_dim,
    #             kernel_size=final_kernel_size,
    #             stride=1,
    #             padding=final_kernel_size // 2,
    #             groups=7
    #         ),
    #         torch.nn.ReLU(),
    #         torch.nn.Conv1d(
    #             in_channels=feature_dim,
    #             out_channels=feature_dim,
    #             kernel_size=final_kernel_size,
    #             stride=1,
    #             padding=final_kernel_size // 2,
    #             groups=feature_dim
    #         ),
    #         torch.nn.ReLU6()
    #     )

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
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                in_channels=in_channels // 8,
                out_channels=in_channels // 16,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
            torch.nn.ReLU()
        )

    # def forward(self, x):
    #     if self.make_4d:
    #         x = x.view(x.size(0), x.size(3), x.size(2))

    #     if self.verbose:
    #         print('\nx\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
    #             torch.mean(x), torch.std(x), torch.min(x), torch.max(x), x.size()))

    #     inp = x.transpose(1, 2)
    #     if self.verbose:
    #         print('\ninp\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
    #             torch.mean(inp), torch.std(inp), torch.min(inp), torch.max(inp), inp.size()))

    #     out1 = self.conv1(inp)
    #     if self.verbose:
    #         print('\nout1\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
    #             torch.mean(out1), torch.std(out1), torch.min(out1), torch.max(out1), out1.size()))

    #     out2 = self.conv2(inp)
    #     if self.verbose:
    #         print('\nout2\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
    #             torch.mean(out2), torch.std(out2), torch.min(out2), torch.max(out2), out2.size()))

    #     out3 = self.conv3(inp)
    #     if self.verbose:
    #         print('\nout3\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
    #             torch.mean(out3), torch.std(out3), torch.min(out3), torch.max(out3), out3.size()))

    #     out4 = self.conv4(inp)
    #     if self.verbose:
    #         print('\nout4\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
    #             torch.mean(out4), torch.std(out4), torch.min(out4), torch.max(out4), out4.size()))

    #     out5 = self.conv5(inp)
    #     if self.verbose:
    #         print('\nout5\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
    #             torch.mean(out5), torch.std(out5), torch.min(out5), torch.max(out5), out5.size()))

    #     out6 = self.conv6(inp)
    #     if self.verbose:
    #         print('\nout6\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
    #             torch.mean(out6), torch.std(out6), torch.min(out6), torch.max(out6), out6.size()))

    #     out = out1 + out2 + out3 + out4 + out5 + out6
    #     # stacked = torch.stack((out1, out2, out3, out4, out5, out6), dim=2)
    #     # out = torch.flatten(stacked, start_dim=1, end_dim=2)

    #     if self.verbose:
    #         print('\nsummed\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
    #             torch.mean(out), torch.std(out), torch.min(out), torch.max(out), out.size()))

    #     out = self.upscale_conv(out)

    #     if self.verbose:
    #         print('\nupscale conv\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
    #             torch.mean(out), torch.std(out), torch.min(out), torch.max(out), out.size()))

    #     out += inp

    #     out = self.final_conv(out)

    #     if self.verbose:
    #         print('\nout\tMean: {:.4g} ± {:.4g}\tMin: {:.4g}\tMax: {:.4g}\tSize: {}'.format(
    #             torch.mean(out), torch.std(out), torch.min(out), torch.max(out), out.size()))

    #     out = out.transpose(1, 2)

    #     if self.make_4d:
    #         out = out.reshape(out.size(0), 1, out.size(2), out.size(1))

    #     return out

    def get_side(self, mask, side, inputs):
        with torch.no_grad():
            max_mask_len = torch.max(torch.sum(mask, 1)).int()

            side_inputs = []

            for batch_index, mask_batch in enumerate(mask):
                mask_len = torch.sum(mask_batch).int()

                if mask_len == 0:
                    continue

                m_nonzero = mask_batch.nonzero().flatten()
                first_nonzero = m_nonzero[0]
                last_nonzero = m_nonzero[-1]

                if side == 'left':
                    end_index = first_nonzero - 1
                    end_index = 10
                    start_index = max(0, end_index - max_mask_len)
                elif side == 'right':
                    start_index = last_nonzero + 1
                    end_index = min(start_index + max_mask_len,
                                    inputs[batch_index].size(1))


                side_input = inputs[batch_index][start_index:end_index]

                if end_index - start_index < max_mask_len:
                    pad_zeros = torch.zeros(
                        (max_mask_len - side_input.shape[0], side_input.shape[1])).to(side_input.device).to(mask.device)
                    if side == 'left':
                        side_input = torch.cat((pad_zeros, side_input), 0)
                    elif side == 'right':
                        side_input = torch.cat((side_input, pad_zeros), 0)

                side_inputs.append(side_input)

        return torch.stack(side_inputs)
    
    def get_nonzero_masked_outputs(self, mask, outputs):
        masked_outputs = []
        for batch_index, mask_batch in enumerate(mask):
            mask_len = torch.sum(mask_batch).int()

            if mask_len == 0:
                continue
            masked_outputs.append(outputs[batch_index])
        
        return torch.stack(masked_outputs)



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
