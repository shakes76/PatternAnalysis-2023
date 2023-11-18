"""
Author: Zach Harbutt S4585714
Contains the source code of the components of the ESPCN model.

ref: https://keras.io/examples/vision/super_resolution_sub_pixel/#build-a-model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ESPCN(nn.Module):
    def __init__(self, upscale_factor=4, channels=1):
        super(ESPCN, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=5, padding='same')
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding='same')
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding='same')
        self.conv4 = nn.Conv2d(32, channels * (upscale_factor ** 2), kernel_size=3, padding='same')
        self.pixel = nn.PixelShuffle(upscale_factor)
        
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = self.pixel(self.conv4(out))
        return out
    
