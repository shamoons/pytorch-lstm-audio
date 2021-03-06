import torch

import torch


def cos_mse_similiarity_loss(inp, target):
    cos_loss = 1 - torch.nn.CosineSimilarity(dim=1)(inp + 1, target + 1)
    cos_loss = cos_loss.mean()

    mse_loss = torch.nn.MSELoss(reduction='mean')(inp, target)

    loss = mse_loss + cos_loss
    # loss = cos_loss

    return loss


def loss_fn(inp, target):
    # zeros_sum = (target == 0).sum(dim = 0).float()
    # one_sum = (target == 1).sum(dim = 0).float()

    # pos_weight = zeros_sum / (one_sum + 1e-2)
    # loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
    loss_fn = torch.nn.BCELoss()

    loss = loss_fn(inp, target)

    return loss


# tensor1 = torch.randn(4, 4)
# tensor2 = torch.empty(4, 4).random_(2)

# print(tensor1, tensor1.size())
# print(tensor2, tensor2.size())


# loss_val = loss_fn(tensor1, tensor2)
# print(loss_val)

tensor1 = torch.Tensor([    0.001,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.001,     0.008,     0.037,     0.129,     0.283,
            0.451,     0.574,     0.683,     0.800,     0.894,     0.951,     0.979,     0.991,     0.996,     0.998,     0.999,     1.000,     1.000,     1.000,     1.000,
            1.000,     1.000,     1.000,     1.000,     0.999,     0.999,     0.996,     0.988,     0.963,     0.896,     0.769,     0.602,     0.448,     0.334,     0.231,
            0.138,     0.071,     0.030,     0.011,     0.004,     0.001,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000,
            0.000,     0.000,     0.000,     0.000,     0.000,     0.000,     0.000])

tensor2 = torch.Tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0.001])


torch.set_printoptions(profile='full', precision=3, sci_mode=False, linewidth=180)
tensor3 = torch.round(tensor1)
print(tensor1, tensor1.size())
print(tensor2, tensor2.size())
print(tensor3, tensor3.size())

print(tensor1.nonzero())
print(tensor2.nonzero())
print(tensor3.nonzero())

loss = loss_fn(tensor1, tensor2)
loss_rounded = loss_fn(tensor3, tensor2)

print(loss, loss_rounded)