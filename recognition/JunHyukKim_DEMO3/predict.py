#import modules
import argparse
import os
import random
import numpy as np

import dataset
from modules import UNET

import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms

from dataset import CustomDataset
from torch.utils.data import DataLoader
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from modules import UNET
from utils import ( get_test_loaders,
                    check_accuracy,
                    save_predictions_as_imgs,
                    make_folder_if_not_exists)



def main():
    make_folder_if_not_exists("evaluation_images")
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 2
    IMAGE_HEIGHT = 256  # 1280 originally
    IMAGE_WIDTH = 256  # 1918 originally
    PIN_MEMORY = True
    TEST_IMG_DIR = "data/test_images/"
    TEST_MASK_DIR = "data/test_masks/"
    BATCH_SIZE = 16

    test_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    test_loader = get_test_loaders(
            TEST_IMG_DIR,
            TEST_MASK_DIR,
            BATCH_SIZE,
            test_transforms,
            NUM_WORKERS,
            PIN_MEMORY,
    )

    FILE = "model.pth"
    loaded_model = UNET(3,1,[64,128,256,512]) 
    loaded_model.load_state_dict(torch.load(FILE))
    loaded_model.to(DEVICE)
    loaded_model.eval()
    check_accuracy(test_loader,loaded_model)
    #save_predictions_as_imgs(test_loader,loaded_model)

if __name__ == "__main__":
    main()