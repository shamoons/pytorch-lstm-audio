import torch.nn as nn

def init_weights(model):
    if isinstance(model, (nn.Dropout, nn.Sequential, nn.Tanh, nn.Sigmoid)) or type(model).__name__ == 'MaskingModel':
        return

    nn.init.ones_(model.weight.data)

    if isinstance(model, nn.PReLU):
        return

    if model.bias is not None:
        nn.init.ones_(model.bias.data)