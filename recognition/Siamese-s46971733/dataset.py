# Imports
import time
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

# Rangpur Locations
# TESTIMAGEPATH = '../../groups/comp3710/ADNI/AD_NC/test'
# TRAINIMAGEPATH = '../../groups/comp3710/ADNI/AD_NC/train'

# Local Locations
TESTIMAGEPATH = '../ADNI/AD_NC/test'
TRAINIMAGEPATH = '../ADNI/AD_NC/train'

# Creating Lists of Directories
train_dirs_full = []
train_dirs_AD = os.listdir(TRAINIMAGEPATH + '/AD')
train_dirs_NC = os.listdir(TRAINIMAGEPATH + '/NC')

test_dirs_full = []
test_dirs_AD = os.listdir(TESTIMAGEPATH + '/AD')
test_dirs_NC = os.listdir(TESTIMAGEPATH + '/NC')

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

#print(train_dirs_full)

