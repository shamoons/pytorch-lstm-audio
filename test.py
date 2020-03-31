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
    print(loss_fn)

    loss = loss_fn(inp, target)

    return loss

tensor1 = torch.randn(4, 4)
tensor2 = torch.empty(4, 4).random_(2)

print(tensor1, tensor1.size())
print(tensor2, tensor2.size())


loss_val = loss_fn(tensor1, tensor2)
print(loss_val)
