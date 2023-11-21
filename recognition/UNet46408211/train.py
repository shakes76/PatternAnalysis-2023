"""
This file is the main training script for the Improved UNet model
"""

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from modules import ImprovedUNet
from dataset import ISICDataset
import time
from utils import *
from global_params import * # Hyperparameters and other global variables
from dice_loss import BinaryDiceLoss

# set the device to cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LOAD_MODEL = False
SAVE_EPOCH_DATA = False#True

def train_epoch(loader, model, optimizer, loss_fn, scaler, losses, train_dice_scores):
    """Trains the model for one epoch

    Args:
        loader (DataLoader): The data loader
        model (nn.Module): The model
        optimizer (torch.optim): The optimizer
        loss_fn (nn.Module): The loss function
        scaler (torch.cuda.amp.GradScaler): The gradient scaler
        losses (list): The list to store the losses
        train_dice_scores (list): The list to store the training dice scores
    """
    loop = tqdm(loader)
    # loop = loader
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.float().unsqueeze(1).to(device=device)
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
        dice = 1 - loss.item() # dice loss = 1 - dice score
        train_dice_scores.append(dice)
        
        # update tqdm loop
        loop.set_postfix(loss=loss.item(), dice_score=dice)

def main():
    train_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        # Could add more transforms here
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2()
    ])
    validation_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2()
    ])
    
    # create the dataloaders 
    train_set = ISICDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform=train_transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    
    validation_set = ISICDataset(VAL_IMG_DIR, VAL_MASK_DIR, transform=validation_transform)
    val_loader = DataLoader(validation_set, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)
    
    # 3 channels in for RGB images, 1 channel out for binary mask
    model = ImprovedUNet(in_channels=3, out_channels=1).to(device)
    loss_fn = BinaryDiceLoss() # Binary Dice Loss from https://github.com/hubutui/DiceLoss-PyTorch see dice_loss.py
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) # Adam optimizer
    # This learning rate scheduler reduces the learning rate by a factor of 0.1 if the mean epoch loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True)
    
    # load model if LOAD_MODEL is True
    if LOAD_MODEL:
        load_checkpoint(torch.load('checkpoints/checkpoint.pth.tar'), model, optimizer)
    
    scaler = torch.cuda.amp.GradScaler()
    
    losses = [] # all training losses
    dice_scores = [] # for plotting
    train_dice_scores = [] # for plotting
    epoch_losses = [] # average loss for each epoch
    
    model.train()
    
    # Training loop
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        train_epoch_losses = [] # train losses for the epoch
        # Train the model for one epoch
        train_epoch(train_loader, model, optimizer, loss_fn, scaler, train_epoch_losses, train_dice_scores)
        
        # Calculate the average loss for the epoch
        epoch_losses.append(np.mean(train_epoch_losses))
        losses.extend(train_epoch_losses)
        
        # Update the learning rate
        scheduler.step(epoch_losses[-1])
        
        # Calculate the validation dice score after each epoch
        val_dice_score = calc_dice_score(model, val_loader, device=device)
        val_dice_score = np.round(val_dice_score.item(), 4)
        dice_scores.append(val_dice_score)
        print(f'Validation dice score: {val_dice_score}')
            
        # Print some feedback after each epoch
        print_progress(start_time, epoch, NUM_EPOCHS)
        
        if SAVE_EPOCH_DATA:
            # Save some predictions to a folder for visualization
            os.makedirs(f'epoch_data', exist_ok=True)
            os.makedirs(f'epoch_data/epoch_{epoch}', exist_ok=True)
            os.makedirs(f'epoch_data/epoch_{epoch}/checkpoints', exist_ok=True)
            os.makedirs(f'epoch_data/epoch_{epoch}/images', exist_ok=True)
            save_predictions_as_imgs(val_loader, model, 10, folder=f'epoch_data/epoch_{epoch}/images/', device=device)
            # Save a checkpoint after each epoch
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            save_checkpoint(checkpoint, filename=f'epoch_data/epoch_{epoch}/checkpoints/checkpoint.pth.tar')
        
    # Save a checkpoint after training is complete
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    save_checkpoint(checkpoint)
        
    # Plot the losses
    plt.figure(figsize=(20, 10))
    plt.plot(losses, label='Loss')
    plt.xlabel('Batch #')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.grid(True)
    plt.savefig('save_data/losses.png')
    plt.show()
    
    # plot the training dice scores
    plt.figure(figsize=(20, 10))
    plt.plot(train_dice_scores, label='Dice Score')
    plt.xlabel('Batch #')
    plt.ylabel('Dice Score')
    plt.grid(True)
    plt.title('Training Dice Scores')
    plt.savefig('save_data/train_dice_scores.png')
    plt.show()
    
    # plot Average Losses per Epoch
    plt.figure(figsize=(20, 10))
    plt.plot(epoch_losses, label='Loss')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    plt.title('Average Losses per Epoch')
    plt.grid(True)
    plt.savefig('save_data/epoch_losses.png')
    plt.show()
    
    # plot dice score vs epoch
    plt.figure(figsize=(20, 10))
    plt.plot(dice_scores, label='Dice Score')
    plt.xlabel('Epoch #')
    plt.ylabel('Dice Score')
    plt.title('Validation Dice Scores')
    plt.grid(True)
    plt.savefig('save_data/dice_scores.png')
    plt.show()

if __name__ == '__main__':
    main()