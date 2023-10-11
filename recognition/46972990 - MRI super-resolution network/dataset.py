import os
from modules import SuperResolutionDataset
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# Paths to the images
BASE_PATH = "C:\\Users\\User\\OneDrive\\Bachelor of Computer Science\\Semester 6 2023\\COMP3710\\ADNI_AD_NC_2D\\AD_NC"
TRAIN_PATH = os.path.join(BASE_PATH, "train")
TEST_PATH = os.path.join(BASE_PATH, "test")

BATCH_SIZE = 32

def get_train_and_validation_loaders():
    full_dataset = SuperResolutionDataset(TRAIN_PATH)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size) #80% train and 20% validation
    val_size = total_size - train_size

    torch.manual_seed(42)
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, validation_loader

def get_test_loader():
    dataset = SuperResolutionDataset(TEST_PATH)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    return data_loader