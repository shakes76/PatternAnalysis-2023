from __future__ import print_function

'''Containing the data loader for loading and preprocessing your data'''

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

#from utils import *

# parsers

parser = argparse.ArgumentParser(description='ADNI Training')
parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
parser.add_argument('--opt', default="adam")
parser.add_argument('--net', default='CCT')
parser.add_argument('--bs', default='32')
parser.add_argument('--size', default="256")
parser.add_argument('--n_epochs', type=int, default='100')
parser.add_argument('--patch', default='128', type=int, help="patch for ViT")
parser.add_argument('--dimhead', default="512", type=int)
classes = ('NC', 'AD')
num_classes = len(classes)


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

# Normalize
def get_mean_std(loader):
    # Compute the mean and standard deviation of all pixels in the dataset
    #loader = torch.utils.data.DataLoader(loader, batch_size=128, shuffle=False)  # Make sure shuffle is set to False
    mean = 0.
    std = 0.
    for images, _ in loader:
        batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    var = 0.0
    pixel_count = 0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1))**2).sum([0,2])
        pixel_count += images.nelement()
    std1 = torch.sqrt(var / pixel_count)
    print(mean, std, std1)
    return mean, std, std1
# Data
print('==> Preparing data..') 
def normalize_train():
    transform_train = transforms.Compose([
    transforms.Resize((size,size)),
    transforms.ToTensor(),
])
    trainset = torchvision.datasets.ImageFolder(root='/home/groups/comp3710/ADNI/AD_NC/train', transform=transform_train)
    #torchvision.datasets.ImageFolder(root='/home/Student/s4737925/Project/Dataset/ADNI_AD_NC_2D/AD_NC/train', transform=transform_train)
    #trainset = torchvision.datasets.ImageFolder(root='Z:/Project/Dataset/ADNI_AD_NC_2D/AD_NC/train', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True)   
    mean, std, std1 = get_mean_std(trainloader) 
    return mean,std, std1
def normalize_test():
    transform_test = transforms.Compose([
    transforms.Resize((size,size)),
    transforms.ToTensor(),
])
    testset = torchvision.datasets.ImageFolder(root='/home/groups/comp3710/ADNI/AD_NC/test', transform=transform_test)
    #torchvision.datasets.ImageFolder(root='/home/Student/s4737925/Project/Dataset/ADNI_AD_NC_2D/AD_NC/train', transform=transform_train)
    #trainset = torchvision.datasets.ImageFolder(root='Z:/Project/Dataset/ADNI_AD_NC_2D/AD_NC/train', transform=transform_train)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)   
    mean, std, std1 = get_mean_std(testloader) 
    return mean, std, std1      

# mean, std, std1 = normalize_train()
# mean_test, std_test, std2 = normalize_test()
# print("Normalization", mean, std, std1)
# print("Normalization Test", mean_test, std_test, std2)
transform_train = transforms.Compose([
    transforms.Resize((size,size)),
    RandAugment(num_ops=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))    
    
    # transforms.RandomRotation(10),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomCrop(size),
    # TrivialAugmentWide(),
    #transforms.RandomResizedCrop(size),
    #transforms.RandomVerticalFlip(),
    # transforms.Normalize(mean=mean, std=std),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
])

transform_test = transforms.Compose([
    transforms.Resize((size,size)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # transforms.Normalize(mean=0.1167, std=0.1297)
    # transforms.Normalize(mean=mean_test, std=std_test),
])

transform_valid = transforms.Compose([
    transforms.Resize((size,size)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # transforms.Normalize(mean=0.1167, std=0.1297)
    # transforms.Normalize(mean=mean_test, std=std_test),
])

# Prepare dataset
trainset = torchvision.datasets.ImageFolder(root='/home/Student/s4737925/Project/Patient_Split/train', transform=transform_train)
#torchvision.datasets.ImageFolder(root='/home/Student/s4737925/Project/Dataset/ADNI_AD_NC_2D/AD_NC/train', transform=transform_train)
#trainset = torchvision.datasets.ImageFolder(root='Z:/Project/Dataset/ADNI_AD_NC_2D/AD_NC/train', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True)

## Test
# testset = torchvision.datasets.ImageFolder(root='/home/groups/comp3710/ADNI/AD_NC/test', transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

## Vaid
testset = torchvision.datasets.ImageFolder(root='/home/Student/s4737925/Project/Patient_Split/test', transform=transform_test)
validset = torchvision.datasets.ImageFolder(root='/home/Student/s4737925/Project/Patient_Split/valid', transform=transform_valid)
#test_valid_set = torchvision.datasets.ImageFolder(root='/home/Student/s4737925/Project/Dataset/ADNI_AD_NC_2D/AD_NC/test', transform=transform_test)
#test_valid_set = torchvision.datasets.ImageFolder(root='Z:/Project/Dataset/ADNI_AD_NC_2D/AD_NC/test', transform=transform_test)

# test_size = int(0.5 * len(test_valid_set))
# valid_size = len(test_valid_set) - test_size

# testset, validset = random_split(test_valid_set, [test_size, valid_size])

testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
validloader = torch.utils.data.DataLoader(validset, batch_size=100, shuffle=False)