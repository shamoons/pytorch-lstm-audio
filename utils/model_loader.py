import glob
import importlib
import json
import sys
import torch
import os.path as path


def load_masking_model(wandb_id, device, make_4d=False):
    wandb_dir = list(glob.iglob(
        path.join('wandb', '*' + wandb_id), recursive=False))[0]
    model_path = path.join(wandb_dir, 'best-model.pt')

    (head, tail) = path.split(model_path)
    mask_args_path = path.join(
        head, tail.replace('best-model.pt', 'args.json'))
    masked_args = json.loads(open(mask_args_path, 'r').read())

    saved_masking_model_filepath = path.join(
        path.abspath(head), 'saved_masking_model.py')
    spec = importlib.util.spec_from_file_location(
        "MaskingModel", saved_masking_model_filepath)
    model_spec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_spec)

    # print(saved_masking_model_filepath)
    # print(spec)
    # print(model_spec)
    model = model_spec.MaskingModel(
        feature_dim=161, kernel_size=masked_args['kernel_size'], kernel_size_step=masked_args['kernel_size_step'], final_kernel_size=masked_args['final_kernel_size'], device=device, make_4d=make_4d)



    # quit()

    # model = importlib.import_module('saved_masking_model').MaskingModel(
    #     feature_dim=161, kernel_size=masked_args['kernel_size'], kernel_size_step=masked_args['kernel_size_step'], final_kernel_size=masked_args['final_kernel_size'], device=device, make_4d=make_4d)
    # sys.path.append(path.abspath(head))
    # print(path.abspath(head))

    # model = importlib.import_module('saved_masking_model').MaskingModel(
    #     feature_dim=161, kernel_size=masked_args['kernel_size'], kernel_size_step=masked_args['kernel_size_step'], final_kernel_size=masked_args['final_kernel_size'], device=device, make_4d=make_4d)

    state_dict = torch.load(model_path, map_location=device)

    model.load_state_dict(state_dict)

    model = model.float()
    model.eval()

    return model
