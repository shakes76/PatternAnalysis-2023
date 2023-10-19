"""
File containing the module/s used in the efficient sup pixel convolutional neural network.
"""
import torch.nn as nn
from utils import *

class Network(nn.Module):
    def __init__(self, upscale_factor=upscale_factor, channels=1, dropout_probability=0.3):
        super(Network, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_probability)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout_probability)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout_probability)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_probability)
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, channels * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        out = self.pixel_shuffle(x5)
        return out
