import numpy as np
import torch.nn as nn
from PIL import Image
import torch
import matplotlib.pyplot as plt
from dataset import ISICDataset
from torch.utils.data import DataLoader
import os
import time

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    print('>>> Saving checkpoint')
    torch.save(state, filename)
    print('>>> Checkpoint saved')

def load_checkpoint(checkpoint, model, optimizer):
    print('>>> Loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print('>>> Checkpoint loaded')

def create_dataloader(img_dir, mask_dir, transform, 
                      batch_size, num_workers, 
                      pin_memory, shuffle=True):
    
    dataset = ISICDataset(img_dir, mask_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, 
                        num_workers=num_workers, 
                        pin_memory=pin_memory, shuffle=shuffle)
    return loader

def dice_score(preds, targets, smooth=1e-6):
    """
    Dice score is the F1 score for binary classification problems.
    """
    preds = preds.float()
    targets = targets.float()
    intersection = 2 * (preds * targets).sum()
    union = preds.sum() + targets.sum()
    return (intersection + smooth) / (union + smooth)

def calc_dice_score(model, dataloader, device='cuda'):
    
    model.eval()
    
    with torch.no_grad():
        dice_score = 0
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            preds = (preds > 0.5).float()
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
    model.train()
    return dice_score / len(dataloader)

def plot_samples_mask_overlay(dataset, n=12):
    """
    Plots n samples from the dataset
    """
    fig, axs = plt.subplots(2, n//2, figsize=(20, 10))
    for i in range(n):
        img, mask = dataset[i]
        axs[i//6, i%6].imshow(img)
        axs[i//6, i%6].imshow(mask, alpha=0.3) # overlay mask
        axs[i//6, i%6].axis('off')
        axs[i//6, i%6].set_title('Sample #{}'.format(i))
    plt.show()

def print_progress(start_time, epoch, num_epochs):
    """
    Estimates the time remaining in the training loop and prints the progress
    """
    elapsed_time = time.time() - start_time
    average_time_per_epoch = elapsed_time / (epoch + 1)
    remaining_time = average_time_per_epoch * (NUM_EPOCHS - epoch - 1)
    # convert to days, hours, minutes, seconds
    days = remaining_time // (24 * 3600)
    remaining_time = remaining_time % (24 * 3600)
    hours = remaining_time // 3600
    remaining_time %= 3600
    minutes = remaining_time // 60
    remaining_time %= 60
    seconds = remaining_time
    print(f'Epoch [{epoch+1}/{num_epochs}] completed. Time elapsed: {elapsed_time:.2f}\
        seconds. Time remaining: {days:.0f} days, {hours:.0f} hours, \
            {minutes:.0f} minutes, {seconds:.2f} seconds')


# # test plot_samples and ISICDataset
# TRAIN_IMG_DIR = 'data/ISIC_2017/Training/ISIC-2017_Training_Data'
# TRAIN_MASK_DIR = 'data/ISIC_2017/Training/ISIC-2017_Training_Part1_GroundTruth'

# isic_data = ISICDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR)
# plot_samples_mask_overlay(isic_data)