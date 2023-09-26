import torch
import torch.nn as nn
import torch.nn.functional as F

class SuperResolution(nn.Module):
    def __init__(self):
        super().__init__()
        #In size is 60 64
        self.conv1 = nn.ConvTranspose2d(1, 128, 4, 2, 1, bias=False) #[n, 64, 120,]
        self.batchnorm1 = nn.BatchNorm2d(128)
        self.conv2 = nn.ConvTranspose2d(128, 1, 4, 2, 1, bias=False)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        #x = self.batchnorm1(x)
        x = self.conv2(x)
        return x