import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from modules import UNet
from dataset import ISICDataset
import time
from utils import *

#----------------------------------------------------------------------
# set the device to cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print('No GPU detected. Using CPU instead.')
print('Using device:', device)
#----------------------------------------------------------------------

# Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 1
NUM_WORKERS = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 512
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = 'data/ISIC_2017/Training/ISIC-2017_Training_Data'
TRAIN_MASK_DIR = 'data/ISIC_2017/Training/ISIC-2017_Training_Part1_GroundTruth'
TEST_IMG_DIR = 'data/ISIC_2017/Testing/ISIC-2017_Test_v2_Data'
TEST_MASK_DIR = 'data/ISIC_2017/Testing/ISIC-2017_Test_v2_Part1_GroundTruth'

def train_epoch(loader, model, optimizer, loss_fn, scaler, losses):
    loop = tqdm(loader)
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
        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        
        # loader.set_postfix(loss=loss.item())
        # print(f'Batch {batch_idx} loss: {loss.item()}')

def main():
    # train_transform = transforms.Compose([
    #     transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    #     transforms.ToTensor(),
    # ])
    # test_transform = transforms.Compose([
    #     transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    #     transforms.ToTensor(),
    # ])
    
    train_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        # Could add more transforms here
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2()
    ])
    test_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2()
    ])
    
    
    # create the dataloaders 
    train_set = ISICDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform=train_transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    
    test_set = ISICDataset(TEST_IMG_DIR, TEST_MASK_DIR, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    
    
    # train_loader = create_dataloader(TRAIN_IMG_DIR, TRAIN_MASK_DIR, train_transform, 
    #                                  BATCH_SIZE, NUM_WORKERS, PIN_MEMORY)
    # test_loader = create_dataloader(TEST_IMG_DIR, TEST_MASK_DIR, test_transform,
    #                                 BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, shuffle=False)
    
    
    # 3 channels in for RGB images, 1 channel out for binary mask
    model = UNet(in_channels=3, out_channels=1).to(device)
    # loss_fn = nn.BCELoss() # Binary Cross Entropy Loss
    loss_fn = nn.BCEWithLogitsLoss() # Binary Cross Entropy Loss with Logits
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) # Adam optimizer
    
    if LOAD_MODEL:
        load_checkpoint(torch.load('checkpoints/my_checkpoint.pth.tar'), model, optimizer)
    
    scaler = torch.cuda.amp.GradScaler()
    
    losses = [] # for plotting
    dice_scores = [] # for plotting
    
    # load model if LOAD_MODEL is True
    
    # Training loop
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        # Train the model for one epoch
        train_epoch(train_loader, model, optimizer, loss_fn, scaler, losses)
        
        # Save a checkpoint after each epoch
        # checkpoint = {
        #     'state_dict': model.state_dict(),
        #     'optimizer': optimizer.state_dict()
        # }
        # save_checkpoint(checkpoint)
        
        print('>>> Calculating Epoch Dice Score')
        dice_score = calc_dice_score(model, test_loader, device=device)
        dice_scores.append(dice_score)
        print(f'Dice score: {dice_score}')
        
        # Save some predictions to a folder for visualization
        # FOLDER NAME SHOULD BE BASED ON THE CURRENT EPOCH
        
        # Print some feedback after each epoch
        print_progress(start_time, epoch, NUM_EPOCHS)
        
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
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()