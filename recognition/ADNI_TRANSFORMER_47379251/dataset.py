from __future__ import print_function ## Python specifies this to be added in the first line

''' This file is used for loading and preprocessing training data, validation data and test data
please refer to read me and the code for more insights'''

import os
import argparse
import csv
import time
import random
import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torchvision.transforms import RandAugment
from torchvision.transforms import TrivialAugmentWide
# from util import normalize_train, normalize_test

# Initializing variables

classes = ('NC', 'AD')
num_classes = len(classes)
bs = 32 # Batch size
imsize = 256 # Image size

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
size = imsize


# Data
print('==> Preparing data..')     

# mean, std, std1 = normalize_train()
# mean_test, std_test, std2 = normalize_test()
# print("Normalization", mean, std, std1)
# print("Normalization Test", mean_test, std_test, std2)

transform_train = transforms.Compose([
    transforms.Resize((size,size)),
    RandAugment(num_ops=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))    
])

transform_test = transforms.Compose([
    transforms.Resize((size,size)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_valid = transforms.Compose([
    transforms.Resize((size,size)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Prepare dataset
trainset = torchvision.datasets.ImageFolder(root='/home/Student/s4737925/Project/Patient_Split/train', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True)

## Test & Vaid
testset = torchvision.datasets.ImageFolder(root='/home/Student/s4737925/Project/Patient_Split/test', transform=transform_test)
validset = torchvision.datasets.ImageFolder(root='/home/Student/s4737925/Project/Patient_Split/valid', transform=transform_valid)

testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
validloader = torch.utils.data.DataLoader(validset, batch_size=100, shuffle=False)

print(len(trainset), len(testset), len(validset))