import torch
from torch import nn
from dataset import CustomDataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
import torch
import torch.nn as nn
import torch.nn.functional as funk
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from modules import UNet

##init
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device='cpu'
if not torch.cuda.is_available():
    print("Warning no cuda")

## Data
transform = transforms.Compose([transforms.Resize((512, 512)),transforms.ToTensor()])

trainset = CustomDataset(root_dir='/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Training_Input_x2', transform=transform) 
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
total_step=len(trainloader)
print(total_step)

def improved_UNet():
    return UNet()

model=improved_UNet()
model=model.to(device)

for images in trainloader:
    images=images.to(device)
    output=model(images)
    
