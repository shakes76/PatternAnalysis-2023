import torch
import torch.nn as nn
import torch.optim as optim

import CONSTANTS

def load_from_checkpoint(filename:str, model:nn.Module, optimizer:optim.Optimizer):
    """
    Loads a model from a checkpoint file.
    args:
        filename: the name of the checkpoint file only. Do not include the path.
            the filepath should be defined in CONSTANTS.py
        model: the model to load the checkpoint into
        optimizer: the optimizer to load the checkpoint into
    """
    checkpoint = torch.load(CONSTANTS.MODEL_PATH + filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    starting_epoch = checkpoint['epoch']
    training_loss = checkpoint['loss_train']
    eval_loss = checkpoint['loss_eval']
    print(f"Loading {model.__class__.__name__}. Last trained epoch: {str(starting_epoch)}")
    return starting_epoch, model, optimizer, training_loss, eval_loss

def save_checkpoint(epoch:int, model:nn.Module, optimizer:optim.Optimizer, training_loss:list, eval_loss:list):
    """
    Saves a model to a checkpoint file.
    args:
        epoch: the next epoch to be trained
        model: the model to save
        optimizer: the optimizer to save
        training_loss: a list of training losses in the epochs so far
        eval_loss: a list of validation losses in the epochs so far
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_train': training_loss,
        'loss_eval': eval_loss
    }, CONSTANTS.RESULTS_PATH + f"{model.__class__.__name__}_checkpoint.tar"
    )