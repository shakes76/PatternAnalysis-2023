import os
import numpy as np
import random
from torchvision import datasets, transforms
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

ROOT_DIR = "/home/groups/comp3710/ADNI/AD_NC/train"

def get_transforms():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Adjust these values if needed, for 1 channel
    ])


train_dataset = ImageFolder(root=ROOT_DIR, transform=get_transforms())

print(train_dataset.classes)  # ['AD', 'NC'] or ['NC', 'AD'] depending on folder ordering
print(train_dataset.class_to_idx)  # {'AD': 0, 'NC': 1} or {'NC': 0, 'AD': 1}

# one image
image, label = train_dataset[0]

# see class distribution
class_counts = {}
# Iterate over each class index and count occurrences in `train_dataset.targets`
for idx, class_name in enumerate(train_dataset.classes):
    class_counts[class_name] = train_dataset.targets.count(idx)

print("Total Number of images:",  len(train_dataset))
print("Class Counts:",class_counts)
print("Image Size:", image.shape)


