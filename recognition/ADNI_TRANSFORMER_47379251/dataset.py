from __future__ import print_function
'''Containing the data loader for loading and preprocessing your data'''
'''

Train CIFAR10 with PyTorch and Vision Transformers!
written by @kentaroy47, @arutema47

'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pandas as pd
import csv
import time

from modules import *
from utils import *

# parsers
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--opt', default="adam")
# parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
# parser.add_argument('--noaug', action='store_true', help='disable use randomaug')
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
# parser.add_argument('--nowandb', action='store_true', help='disable wandb')
# parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='vit')
parser.add_argument('--bs', default='16')
parser.add_argument('--size', default="256")
parser.add_argument('--n_epochs', type=int, default='5')
parser.add_argument('--patch', default='32', type=int, help="patch for ViT")
parser.add_argument('--dimhead', default="1024", type=int)
parser.add_argument('--convkernel', default='4', type=int, help="parameter for convmixer")

args = parser.parse_args()

# take in args
# usewandb = ~args.nowandb
# if usewandb:
#     import wandb
#     watermark = "{}_lr{}".format(args.net, args.lr)
#     wandb.init(project="cifar10-challange",
#             name=watermark)
#     wandb.config.update(args)

bs = int(args.bs)
imsize = int(args.size)

use_amp = not args.noamp
# aug = args.noaug

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
# if args.net=="vit_timm":
#     size = 384
# else:
size = imsize    

transform_train = transforms.Compose([
    transforms.Resize(size),
    transforms.RandomCrop(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Add RandAugment with N, M(hyperparameter)
# if aug:  
#     N = 2; M = 14;
#     transform_train.transforms.insert(0, RandAugment(N, M))

# Prepare dataset
trainset = torchvision.datasets.ImageFolder(root='/home/Student/s4737925/Project/Dataset/ADNI_AD_NC_2D/AD_NC/train', transform=transform_train)
# trainset = torchvision.datasets.ImageFolder(root='Z:/Project/Dataset/ADNI_AD_NC_2D/AD_NC/train', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True)

testset = torchvision.datasets.ImageFolder(root='/home/Student/s4737925/Project/Dataset/ADNI_AD_NC_2D/AD_NC/test', transform=transform_test)
# testset = torchvision.datasets.ImageFolder(root='Z:/Project/Dataset/ADNI_AD_NC_2D/AD_NC/test', transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False)

classes = ('NC', 'AD')