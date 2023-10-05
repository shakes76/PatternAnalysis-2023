import torch.nn as nn
import os
import numpy as np
import random
from torchvision import datasets, transforms
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from PIL import Image
import torch.optim as optim
import torch.nn.functional as F

from dataset import create_siamese_dataloader,get_transforms_training, get_transforms_testing
from modules import SiameseNetwork




print("------Testing---------")
ROOT_DIR_TEST = "/home/groups/comp3710/ADNI/AD_NC/test"
test_loader = create_siamese_dataloader(ROOT_DIR_TEST, batch_size=32, transform=get_transforms_testing())

# h, w = 240, 256
# input_shape=(1, h, w)
model = SiameseNetwork()

# Load the saved model weights
model_path = '/home/Student/s4757184/Pattern_Project/PatternAnalysis-2023/recognition/47571840-Adni-siamese/model_3_20epoch.pth'
model.load_state_dict(torch.load(model_path))


# Move the model to the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# # Your testing loop
# correct_pairs = 0
# total_pairs = 0

# with torch.no_grad():
#     for img1, img2, labels in test_loader:
#         img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
#         outputs = model(img1, img2)
#         predicted = (outputs > 0.5).float().squeeze()  # Ensure squeezing if necessary
#         correct_pairs += (predicted == labels).sum().item()
#         total_pairs += labels.size(0)

# accuracy = 100.0 * correct_pairs / total_pairs
# print(f"Accuracy on test data: {accuracy:.2f}%")



# If you've saved the model
# model = SiameseNetwork()
# model.load_state_dict(torch.load("path_to_saved_model.pth"))
# model.to(device)

model.eval()  # Set the model to evaluation mode

correct = 0
total = 0

with torch.no_grad():  # No need to compute gradients during evaluation
    for img1, img2, labels in test_loader:
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

        outputs = model(img1, img2).squeeze()
        preds = (outputs >= 0.5).float().squeeze()
        
        total += labels.size(0)
        correct += (preds == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy on test data: {accuracy:.2f}%")

