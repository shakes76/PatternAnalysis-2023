"""
Reference: https://keras.io/examples/vision/super_resolution_sub_pixel/
"""

import torch.nn as nn
import torch.nn.functional as F
from utils import *

class SuperResolution(nn.Module):
    def __init__(self, upscale_factor=4, channels=1):
        super(SuperResolution, self).__init__()

        self.conv1 = nn.Conv2d(channels, 64, 5, padding=2)
        self.conv2 = nn.Conv2d(64, out_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv4 = nn.Conv2d(out_channels, channels * (upscale_factor ** 2), 
                               3, padding=1)

        #efficient sub-pixel layer
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.pixel_shuffle(self.conv4(x), upscale_factor=4)

        return x

