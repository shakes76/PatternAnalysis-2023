from dataset import ISICDataset
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import time
from utils import *
from modules import UNet
from dataset import ISICDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

#----------------------------------------------------------------------
# set the device to cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# if not torch.cuda.is_available():
#     print('No GPU detected. Using CPU instead.')
# print('Using device:', device)
#----------------------------------------------------------------------

# Test data
TEST_IMG_DIR = 'data/ISIC_2017/Testing/ISIC-2017_Test_v2_Data'
TEST_MASK_DIR = 'data/ISIC_2017/Testing/ISIC-2017_Test_v2_Part1_GroundTruth'
# Validation data
VAL_IMG_DIR = 'data/ISIC_2017/Validation/ISIC-2017_Validation_Data'
VAL_MASK_DIR = 'data/ISIC_2017/Validation/ISIC-2017_Validation_Part1_GroundTruth'
# Hyperparameters
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
BATCH_SIZE = 1 # We want to test one image at a time
NUM_WORKERS = 1
CHECKPOINT_DIR = 'old/epoch_data1/epoch_19/checkpoints/checkpoint.pth.tar'

def main():
    model = UNet(in_channels=3, out_channels=1).to(device)
    load_checkpoint(torch.load(CHECKPOINT_DIR), model)

    # Defining the test loader
    test_transform = A.Compose([
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            ToTensorV2()
        ])
    test_set = ISICDataset(TEST_IMG_DIR, TEST_MASK_DIR, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # Defining the validation loader
    val_set = ISICDataset(VAL_IMG_DIR, VAL_MASK_DIR, transform=test_transform)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # calculate the dice score on the test set
    dice_score = calc_dice_score(model, test_loader, device=device, verbose=True)
    print(f'Test Dice Score: {dice_score:.4f}')

    # print('>>> Generating and saving predictions')
    # folder = 'old/epoch_data1/epoch_18/images/'
    save_predictions_as_imgs(test_loader, model, num=6, folder='saved_images/', device=device)

    # print('>>> Predictions saved')
    plot_samples(6, title='Epoch 18')
    # plot_prediction()

if __name__ == '__main__':
    main()