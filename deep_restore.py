import torch
from utils.model_loader import load_masking_model, load_reconstruction_model

class DeepRestore:
    def __init__(self, mask_wandb, reconstruct_wandb, device):
        torch.manual_seed(0)

        self.mask_model = load_masking_model(mask_wandb, device)
        self.reconstruct_model = load_reconstruction_model(reconstruct_wandb, device)