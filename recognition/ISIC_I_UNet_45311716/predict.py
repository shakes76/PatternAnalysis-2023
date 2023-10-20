import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import matplotlib.pyplot as plt
from modules import ImprovedUNet
from dataset import UNetData

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# Path to images
path = 'C:/Users/mlamt/OneDrive/UNI/2023/Semester 2/COMP3710/Code/data/ISIC2018/'
model_path = 'C:/Users/mlamt/OneDrive/UNI/2023/Semester 2/COMP3710/Code/data/ImprovedUNet.pt'
image_path = 'C:/Users/mlamt/OneDrive/UNI/2023/Semester 2/COMP3710/Code/data/images/'

# Hyper-parameters
num_epochs = 15
learning_rate = 1e-3
image_height = 512 
image_width = 512
batch_size = 16

# Following function is from github:
# Reference: https://github.com/pytorch/pytorch/issues/1249#issuecomment-305088398
def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

# Load trained model
model = torch.load(model_path)
# Get test data loader
test_data = UNetData.get_test_loader()

print(' - - Start Predictions - - ')
model.eval()
with torch.no_grad:
    for i, (image, mask) in enumerate(test_data):
        image = image.to(device)
        mask = mask.to(device)

        pred = model(image)

        torchvision.utils.save_image(pred, f"{image_path}Prediction_{i}.png")
        torchvision.utils.save_image(mask, f"{image_path}Actual_{i}.png")
        
        dice_score = 1 - dice_loss(pred, mask)

        print(f"Dice Score of Prediction {i}: " + "{:.4f}".format(dice_score))

        if i > 4:
            break
