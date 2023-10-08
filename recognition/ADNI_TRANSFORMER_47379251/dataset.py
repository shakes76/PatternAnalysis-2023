from __future__ import print_function

'''Containing the data loader for loading and preprocessing your data'''

import os
import argparse
import csv
import time
import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from modules import *
from utils import *

# parsers
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', default=3e-4, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--opt', default="adamw")
parser.add_argument('--net', default='SViT')
parser.add_argument('--bs', default='64')
parser.add_argument('--size', default="256")
parser.add_argument('--n_epochs', type=int, default='5')
parser.add_argument('--patch', default='64', type=int, help="patch for ViT")
parser.add_argument('--dimhead', default="8", type=int)
# parser.add_argument('--convkernel', default='2', type=int, help="parameter for convmixer")
# parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
# parser.add_argument('--noaug', action='store_true', help='disable use randomaug')
# parser.add_argument('--nowandb', action='store_true', help='disable wandb')
# parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
# parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')

args = parser.parse_args()


bs = int(args.bs)
imsize = int(args.size)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
size = imsize
classes = ('NC', 'AD')
num_classes = len(classes)

# Data
print('==> Preparing data..')    

transform_train = transforms.Compose([
    transforms.Resize((size,size)),
    transforms.RandomCrop(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
])

transform_test = transforms.Compose([
    transforms.Resize((size,size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
])

# Prepare dataset
#trainset = torchvision.datasets.ImageFolder(root='/home/Student/s4737925/Project/Dataset/ADNI_AD_NC_2D/AD_NC/train', transform=transform_train)
trainset = torchvision.datasets.ImageFolder(root='Z:/Project/Dataset/ADNI_AD_NC_2D/AD_NC/train', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True)

#testset = torchvision.datasets.ImageFolder(root='/home/Student/s4737925/Project/Dataset/ADNI_AD_NC_2D/AD_NC/test', transform=transform_test)
testset = torchvision.datasets.ImageFolder(root='Z:/Project/Dataset/ADNI_AD_NC_2D/AD_NC/test', transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
