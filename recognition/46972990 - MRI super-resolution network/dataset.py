import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Paths to the images
BASE_PATH = "C:\\Users\\User\\OneDrive\\Bachelor of Computer Science\\Semester 6 2023\\COMP3710\\ADNI_AD_NC_2D\\AD_NC"
TRAIN_PATH = os.path.join(BASE_PATH, "train")
TEST_PATH = os.path.join(BASE_PATH, "test")

# Image size and batch size
IMG_WIDTH = 256
IMG_HEIGHT = 240
BATCH_SIZE = 32