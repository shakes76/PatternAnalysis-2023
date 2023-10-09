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

from dataset import get_dataset
from modules import Resnet, Resnet34, classifer

# Path that model is saved to and loaded from.
PATH = './resnet_net.pth'
CLAS_PATH = './clas_net.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("No CUDA Found. Using CPU")

print("\n")

resnet = Resnet().to(device)
clas_net = classifer().to(device)

batch_size = 6

resnet.load_state_dict(torch.load(PATH))
clas_net.load_state_dict(torch.load(CLAS_PATH))

# Datasets and Dataloaders
testset = get_dataset(train=0, clas=0)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)