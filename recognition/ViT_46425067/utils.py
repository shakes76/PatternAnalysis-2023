"""
Created on Wednesday September 20 2023

Scripts for misc functions used in training, saving and loading the model

@author: Rodger Xiang s4642506
"""
import torch
from pathlib import Path

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
    MODEL_PATH = Path(f"./models/{model_name}.pth")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    torch.save(obj=model.state_dict(), f=MODEL_PATH)
    

def load_model(path, model, device):
    """loads the weights onto a pre-initialised model

    Args:
        path (Path): path to the saved model weights
        model (nn.Module): initialised model with the same parameters that the saved model weights used
        device (str): load model onto CUDA or cpu 

    Returns:
        _type_: _description_
    """
    model.load_state_dict(torch.load(f=path))
    return model.to(device)
