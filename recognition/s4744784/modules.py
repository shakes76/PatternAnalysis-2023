"""
File containing the module/s used in the efficient sup pixel convolutional neural network.
"""
import torch.nn as nn
import torch.nn.functional as F
from utils import *

class Network(nn.Module):
    def __init__(self, upscale_factor=upscale_factor, channels=1):
        super(Network, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, channels * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.ReLU())
        
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        out = self.pixel_shuffle(x4)
        return out
