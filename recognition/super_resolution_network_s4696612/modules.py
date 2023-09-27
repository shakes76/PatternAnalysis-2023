import torch
import torch.nn as nn
import torch.nn.functional as F
6
class SuperResolution(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1 = nn.Upsample(scale_factor=4, mode="bicubic")
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=1, padding_mode='replicate')
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=2, padding_mode='replicate')
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, padding=0, padding_mode='replicate')
        
    
    def forward(self, x):
        x = self.up1(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x