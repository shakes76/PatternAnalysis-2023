import torch
import torch.nn as nn
import torch.optim as optim

import CONSTANTS

def load_from_checkpoint(filename:str, model:nn.Module, optimizer:optim.Optimizer):
    checkpoint = torch.load(CONSTANTS.MODEL_PATH + filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    starting_epoch = checkpoint['epoch']
    training_loss = checkpoint['loss_train']
    eval_loss = checkpoint['loss_eval']
    print(f"Resuming {model.__class__.__name__} training from epoch {str(starting_epoch)}")
    return starting_epoch, model, optimizer, training_loss, eval_loss

def save_checkpoint(epoch:int, model:nn.Module, optimizer:optim.Optimizer, training_loss:list, eval_loss:list):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_train': training_loss,
        'loss_eval': eval_loss
    }, CONSTANTS.RESULTS_PATH + f"{model.__class__.__name__}_checkpoint.tar"
    )