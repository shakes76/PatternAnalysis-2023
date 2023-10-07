# Imports
import time
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2
import os

# Rangpur Locations
# TESTIMAGEPATH = '../../groups/comp3710/ADNI/AD_NC/test'
# TRAINIMAGEPATH = '../../groups/comp3710/ADNI/AD_NC/train'

# # Local Locations
# TESTIMAGEPATH = '../ADNI/AD_NC/test'
# TRAINIMAGEPATH = '../ADNI/AD_NC/train'

# Local Locations
TESTIMAGEPATH = '..\\ADNI\\AD_NC\\test'
TRAINIMAGEPATH = '..\\ADNI\\AD_NC\\train'

# Creating Lists of Directories
train_dirs_full = []
train_dirs_AD = os.listdir(TRAINIMAGEPATH + '\\AD')
train_dirs_NC = os.listdir(TRAINIMAGEPATH + '\\NC')

test_dirs_full = []
test_dirs_AD = os.listdir(TESTIMAGEPATH + '\\AD')
test_dirs_NC = os.listdir(TESTIMAGEPATH + '\\NC')

# Appending full image paths.
for i in os.listdir(TRAINIMAGEPATH):
    if i == 'AD':
        for j in train_dirs_AD:
            train_dirs_full.append(os.path.join(TRAINIMAGEPATH, 'AD', j))
    else:
        for j in train_dirs_NC:
            train_dirs_full.append(os.path.join(TRAINIMAGEPATH, 'NC', j))

for i in os.listdir(TESTIMAGEPATH):
    if i == 'AD':
        for j in test_dirs_AD:
            train_dirs_full.append(os.path.join(TESTIMAGEPATH, 'AD', j))
    else:
        for j in test_dirs_NC:
            train_dirs_full.append(os.path.join(TESTIMAGEPATH, 'NC', j))


##### Temp Code for Visualising Images.
# fig, axs = plt.subplots(2)
# image1 = train_dirs_full[1]
#
# data = cv2.imread(image1)
# axs[0].imshow(data)
# data2 = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
# axs[1].imshow(data2)
#
# plt.show()

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.images = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        # Read image and labels.
        data = cv2.imread(img_path)
        label = img_path.split('\\')[-2]

        # Convert image from BGR to RGB colour. # as imread assumes BGR.
        # data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

        # If transforms are given.
        if self.transform is not None:
            # apply the transformations to both image and segmentation
            # e.g. transform [H, W, n] format to [n, H, W]
            image = self.transform(data)
        else:
            image = data

        return image, label

