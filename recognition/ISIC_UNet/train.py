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

##init
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device='cpu'
if not torch.cuda.is_available():
    print("Warning no cuda")

## Data
transform = transforms.Compose([transforms.ToTensor()])

trainset = CustomDataset(root_dir='/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Training_Input_x2', transform=transform) 
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
total_step=len(trainloader)
print(total_step)

class pre_act(nn.Module):
    def __init__(self):
        super(pre_act,self).__init__()
        self.bn1=nn.BatchNorm2d()
        self.conv1=nn.Conv2d(kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d()
        self.conv2=nn.Conv2d(kernel_size=3,stride=1,padding=1,bias=False)

    def forward(self,x):
        out=self.bn1(x)
        out=funk.relu(out)
        out=self.conv1(out)
        out=nn.Dropout(p=0.3)
        out=self.bn2(out)
        out=funk.relu(out)
        out=self.conv2(out)
        return out