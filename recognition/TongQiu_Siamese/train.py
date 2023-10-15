import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from utils import Config
from torch.utils.tensorboard import SummaryWriter

def main(model, train_loader, val_loader, criterion, optimizer, epochs):
    print('---------Train on: ' + Config.DEVICE + '----------')

    # Create model
    model = model.to(Config.DEVICE)
    best_score = 0
    writer = SummaryWriter(log_dir=Config.LOG_DIR)  # for TensorBoard

    for epoch in range(epochs):

        # train
        train_batch_loss, train_batch_acc = train(model, train_loader, optimizer, criterion, epoch, epochs)
        # validate
        val_batch_loss, val_batch_acc = validate(model, val_loader, epoch, epochs)

        if val_batch_acc > best_score:
            print(f"model improved: score {best_score:.5f} --> {val_batch_acc:.5f}")
            best_score = val_batch_acc
            # Save the best weights if the score is improved
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'train_loss': train_batch_loss,
                'val_acc': val_batch_acc
            }, Config.MODEL_DIR)
        else:
            print(f"no improvement: score {best_score:.5f} --> {val_batch_acc:.5f}")

        # Write loss and score to TensorBoard
        writer.add_scalar("Training Loss", train_batch_loss.item(), epoch)
        writer.add_scalar("Training Score", train_batch_acc.item(), epoch)
        writer.add_scalar("Validation Loss", val_batch_loss.item(), epoch)
        writer.add_scalar("Validation Score", val_batch_acc.item(), epoch)

    writer.close()

def train(model, train_loader, optimizer, criterion, epoch, epochs):

    pass


def validate(model, val_loader, epoch, epochs):
    pass