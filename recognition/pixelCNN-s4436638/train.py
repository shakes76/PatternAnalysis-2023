import os
import torch
from torch.utils.data import DataLoader
from modules import pixelCNN
from dataset import GetADNITrain

# Get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Define the path to the dataset
images_path = "/home/Student/s4436638/Datasets/AD_NC/train/*"