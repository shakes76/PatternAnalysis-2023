"""
Scripts for misc functions used in training, saving and loading the model

@author: Rodger Xiang s4642506
"""
import torch
from pathlib import Path
from modules import ViT
import platform
import wandb

OS = platform.system()
if OS == "Windows":
    MODEL_PATH = Path("E:/UNI 2023 SEM 2/COMP3710/Lab3/recognition/ViT_46425067/models")
else:
    MODEL_PATH = Path("./models")

def accuracy(y_pred, y):
    """calculates the average accuracy over a batch given the y logits and y targets

    Args:
        y_pred (torch.Tensor): logit values output from the model
        y (torch.Tensor): truth values

    Returns:
        float: average accuracy over the batch  
    """
    y_pred_label = torch.round(torch.sigmoid(y_pred))
    correct = torch.eq(y, y_pred_label).sum().item()
    acc = (correct / len(y))
    return acc

def save_model(model, model_name):
    """Saves the trained model weights

    Args:
        model (nn.Module): trained model
        model_name (str): name of the model to save as
    """
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    torch.save(obj=model.state_dict(), f=MODEL_PATH / f"{model_name}.pth")
    

def load_model(model_name, device, config):
    """loads the weights onto a pre-initialised model

    Args:
        path (Path): path to the saved model weights
        model (nn.Module): initialised model with the same parameters that the saved model weights used
        device (str): load model onto CUDA or cpu 

    Returns:
        nn.Module: _description_
    """
    model = ViT(img_size=config.img_size,
                    patch_size=config.patch_size,
                    img_channels=config.img_channel,
                    num_classes=config.num_classes,
                    embed_dim=config.embed_dim,
                    depth=config.depth,
                    num_heads=config.num_heads,
                    mlp_dim=config.mlp_dim,
                    drop_prob=config.drop_prob).to(device)
    state_dict = torch.load(f=MODEL_PATH / f"{model_name}.pth")
    model.load_state_dict(state_dict)
    return model.to(device)

def create_new_sweep():
    """initialises a sweep according to the sweep config below

    Returns:
        str: sweep id to use when initialising sweep agents
    """
    sweep_config = {
    "method": "grid",
    "metric": {"goal": "maximize", "name": "test/epoch/acc"},
    }
    param_dict =  {
            'epochs': {
                'value': 35
                },
            'img_channel': {
                'value': 1
                },
            'num_classes': {
                'value': 1
                },
            'batch_size': {
                'value': 512
                },
            'img_size': {
                'values': [128, 256]
                },
            'patch_size': {
                'values': [64, 128]
                },
            'embed_dim': {
                'values': [64, 128, 256]
                },
            'depth': {
                'values': [1, 3, 6]
                },
            'num_heads': {
                'value': 8
                },
            'mlp_dim': {
                'value': 1024
                },
            'drop_prob': {
                'value': [0, 0.1]
                },
            'lr': {
                'value': 0.001
                },
            'optimiser': {
                'value': "SGD"
                },
            'linear_embed': {
                'value': True
                },
            'data_augments': {
                'value': "V_H_FLip_Rotate"
                },
            'weight_decay': {
                'values': [0, 1e-6]
            },
            'mix_precision': {
                'value': True
                },
            'lr_scheduler': {
                'value': True
                },
            'max_lr': {
                'value': 0.1
            }
        }   
    sweep_config['parameters'] = param_dict
    sweep_id  = wandb.sweep(sweep_config, project='ViT_Sweep')
    return sweep_id

