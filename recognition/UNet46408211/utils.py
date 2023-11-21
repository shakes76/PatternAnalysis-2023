"""
Some helper functions for visualization, saving/loading checkpoints and calculating the dice score.
"""

import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
import torchvision
import random
import os
import time
from tqdm import tqdm

def save_checkpoint(state, filename='checkpoints/checkpoint.pth.tar'):
    """
    Saves the model and optimizer state dicts to a checkpoint file
    """
    print('>>> Saving checkpoint')
    # os.makedirs('checkpoints', exist_ok=True)
    torch.save(state, filename)
    print('>>> Checkpoint saved')

def load_checkpoint(checkpoint, model, optimizer=None):
    """
    Loads the model and optimizer state dicts from a checkpoint file
    """
    print('>>> Loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    print('>>> Checkpoint loaded')

def calc_dice_score(model, dataloader, device='cuda', verbose=False):
    """
    Calculates the average dice score over the entire dataloader
    """
    model.eval()
    print('>>> Calculating Dice Score')
    with torch.no_grad():
        dice_score = 0
        for _, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            preds = (preds > 0.5).float() # convert to binary mask
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8) # add 1e-8 to avoid division by 0
    model.train()
    dice_score = dice_score / len(dataloader)
    dice_score = dice_score.item()
    dice_score = np.round(dice_score, 4)
    return dice_score

def save_predictions_as_imgs(loader, model, num, folder='saved_images/', device='cuda', verbose=True):
    """
    Saves the predictions from the model as images in the folder
    """
    preds_path = f'{folder}preds/'
    masks_path = f'{folder}masks/'
    orig_path = f'{folder}orig/'
    os.makedirs(preds_path, exist_ok=True)
    os.makedirs(masks_path, exist_ok=True)
    os.makedirs(orig_path, exist_ok=True)
    
    model.eval()
    print('>>> Generating and saving predictions') if verbose else None
    loop = tqdm(enumerate(loader), total=num, leave=False) if verbose else enumerate(loader)
    with torch.no_grad():
        # for idx, (x, y) in enumerate(loader):
        for idx, (x, y) in loop:
            x = x.to(device)
            y = y.to(device) # add 1 channel to mask
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.6).float()
            torchvision.utils.save_image(preds, f'{preds_path}pred_{idx+1}.png')
            torchvision.utils.save_image(y.unsqueeze(1), f'{masks_path}mask_{idx+1}.png')
            torchvision.utils.save_image(x, f'{orig_path}orig_{idx+1}.png')
            if idx == num-1:
                break
    model.train()

def plot_prediction(ind=0, folder='saved_images'):
    """
    Assumes the folder contains the following subfolders:
    preds, masks, orig.
    Plots the original image at ind, the mask, and the prediction side by side with labels.
    """
    preds_path = f'{folder}/preds/'
    masks_path = f'{folder}/masks/'
    orig_path = f'{folder}/orig/'
    
    preds = os.listdir(preds_path)
    masks = os.listdir(masks_path)
    origs = os.listdir(orig_path)
    
    if ind >= len(preds):
        raise IndexError(f'Index {ind} out of range. There are {len(preds)} predictions.')
    
    # plot only one image from each folder
    pred = Image.open(preds_path + preds[ind])
    mask = Image.open(masks_path + masks[ind])
    orig = Image.open(orig_path + origs[ind])
    
    fig, axs = plt.subplots(1, 3, figsize=(20, 10))
    axs[0].imshow(orig)
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    axs[1].imshow(mask)
    axs[1].set_title('Mask')
    axs[1].axis('off')
    axs[2].imshow(pred)
    axs[2].set_title('Prediction')
    axs[2].axis('off')
    plt.show()

def plot_samples(num, folder='saved_images', include_image=True, shuffle=True, title='Samples'):
    """
    Assumes the folder contains the following subfolders:
    preds, masks, orig.
    Plots num predictions side by side with the masks and original images if include_image=True.
    Similar to plot_prediction but for num images.
    """
    preds_path = f'{folder}/preds/'
    masks_path = f'{folder}/masks/'
    orig_path = f'{folder}/orig/'
    
    preds = os.listdir(preds_path)
    masks = os.listdir(masks_path)
    origs = os.listdir(orig_path)
    
    if num > len(preds):
        raise ValueError(f'num = {num} out of range. There are {len(preds)} predictions.')
    
    if shuffle:
        indices = random.sample(range(len(preds)), num)
    else:
        indices = range(num)
    
    if include_image:
        fig, axs = plt.subplots(num, 3, figsize=(20, 10*num))
        for i, ind in enumerate(indices):
            pred = Image.open(preds_path + preds[ind])
            mask = Image.open(masks_path + masks[ind])
            orig = Image.open(orig_path + origs[ind])
            axs[i, 0].imshow(orig)
            axs[i, 0].set_title('Original Image')
            axs[i, 0].axis('off')
            axs[i, 1].imshow(mask)
            axs[i, 1].set_title('Mask')
            axs[i, 1].axis('off')
            axs[i, 2].imshow(pred)
            axs[i, 2].set_title('Prediction')
            axs[i, 2].axis('off')
        fig.suptitle(title)
        plt.show()
    else:
        fig, axs = plt.subplots(num, 2, figsize=(20, 10*num))
        for i, ind in enumerate(indices):
            pred = Image.open(preds_path + preds[ind])
            mask = Image.open(masks_path + masks[ind])
            axs[i, 0].imshow(mask)
            axs[i, 0].set_title('Mask')
            axs[i, 0].axis('off')
            axs[i, 1].imshow(pred)
            axs[i, 1].set_title('Prediction')
            axs[i, 1].axis('off')
        fig.suptitle(title)
        plt.show()

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
    remaining_time = average_time_per_epoch * (num_epochs - epoch - 1)
    # convert elapsed time to days, hours and minutes
    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    
    # convert to days, hours, minutes, seconds
    days = remaining_time // (24 * 3600)
    remaining_time = remaining_time % (24 * 3600)
    hours = remaining_time // 3600
    remaining_time %= 3600
    minutes = remaining_time // 60
    remaining_time %= 60
    print(f"""Epoch [{epoch+1}/{num_epochs}] completed. Time elapsed: {elapsed_time}. 
          seconds. Time remaining: {days:.0f} days, {hours:.0f} hours, 
          {minutes:.0f} minutes.""")