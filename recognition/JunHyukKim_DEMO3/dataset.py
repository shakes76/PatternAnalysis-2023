#import modules
import argparse
import os
import random
from tkinter.tix import IMAGE
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

#set hyperparamiters
TRAINDATA = "ISIC/ISIC-2017_Training_Data"
TESTDATA = "ISIC/ISIC-2017_Test_v2_Data"
WORKERS = 4
BATCH_SIZE = 128
IMAGE_SIZE = 64
NUMBER_CHANNEL = 3
NUMBER_INPUT = 100
FEATURE_GENERATOR = 64
FEATURE_DISCRIMINATOR = 64
NUM_EPOCHS = 5
LEARNING_RATE = 0.0002
BETA1 = 0.5
NUMBER_GPU = 1

# We can use an image folder dataset the way we have it setup.
# Dataset configureation
dataset = dset.ImageFolder(root=DATAROOT,
                           transform=transforms.Compose([
                               transforms.Resize(IMAGE_SIZE),
                               transforms.CenterCrop(IMAGE_SIZE),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                         shuffle=True, num_workers=WORKERS)
