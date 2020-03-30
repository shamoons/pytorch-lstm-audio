import torch

import torch


def cos_mse_similiarity_loss(inp, target):
    cos_loss = 1 - torch.nn.CosineSimilarity(dim=1)(inp + 1, target + 1)
    cos_loss = cos_loss.mean()

    mse_loss = torch.nn.MSELoss(reduction='mean')(inp, target)

    loss = mse_loss + cos_loss
    # loss = cos_loss

    return loss

tensor1 = torch.randn(10000, 3230)
tensor2 = torch.zeros(tensor1.size())

print(tensor1, tensor1.size())
print(tensor2, tensor2.size())


cos = cos_mse_similiarity_loss(tensor1, tensor2)
print(cos)
