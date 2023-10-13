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
if not torch.cuda.is_available():
    print('No GPU detected. Using CPU instead.')
print('Using device:', device)
#----------------------------------------------------------------------

TEST_IMG_DIR = 'data/ISIC_2017/Testing/ISIC-2017_Test_v2_Data'
TEST_MASK_DIR = 'data/ISIC_2017/Testing/ISIC-2017_Test_v2_Part1_GroundTruth'
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
BATCH_SIZE = 1
NUM_WORKERS = 1

def main():
    model = UNet(in_channels=3, out_channels=1).to(device)

    load_checkpoint(torch.load('checkpoints/checkpoint.pth.tar'), model)

    # Defining the test loader
    test_transform = A.Compose([
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            ToTensorV2()
        ])
    test_set = ISICDataset(TEST_IMG_DIR, TEST_MASK_DIR, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # print('>>> Generating and saving predictions')
    save_predictions_as_imgs(test_loader, model, num=1, folder='saved_images/', device=device)
    print('>>> Predictions saved')
    plot_prediction()

if __name__ == '__main__':
    main()