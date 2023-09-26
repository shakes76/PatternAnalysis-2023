import torch
import torch.nn as nn
import torch.nn.functional as F

class SuperResolution(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(1, 32, (16,17),(1,1),(0,0),(0,0), dilation=2, bias=False) #[n, 128, 120,]
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.ConvTranspose2d(32, 16, (16,17),(1,1),(0,0),(0,0), dilation=2, bias=False) #[n, 128, 120,]
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.conv3 = nn.ConvTranspose2d(16, 8, (16,17),(1,1),(0,0),(0,0), dilation=2, bias=False) #[n, 128, 120,]
        self.batchnorm3 = nn.BatchNorm2d(8)
        self.conv4 = nn.ConvTranspose2d(8, 4, (16,17),(1,1),(0,0),(0,0), dilation=2, bias=False) #[n, 128, 120,]
        self.batchnorm4 = nn.BatchNorm2d(4)
        self.conv5 = nn.ConvTranspose2d(4, 2, (16,17),(1,1),(0,0),(0,0), dilation=2, bias=False) #[n, 128, 120,]
        self.batchnorm5 = nn.BatchNorm2d(2)
        self.conv6 = nn.ConvTranspose2d(2, 1, (16,17),(1,1),(0,0),(0,0), dilation=2, bias=False) #[n, 128, 120,]
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.relu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.relu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.relu(self.conv4(x))
        x = self.batchnorm4(x)
        x = F.relu(self.conv5(x))
        x = self.batchnorm5(x)
        x = self.conv6(x)
        return x