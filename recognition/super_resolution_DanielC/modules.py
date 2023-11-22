"""
Reference: https://keras.io/examples/vision/super_resolution_sub_pixel/
"""

import torch.nn as nn
import torch.nn.functional as F
from utils import *

"""
SuperResolution network per the keras example. The out and in channels of the 
convolutional layers has been modified for improved performance.
"""
class SuperResolution(nn.Module):
    def __init__(self, upscale_factor=4, channels=1):
        """
        Define the layers of the SuperResolution Network.

        Attributes:
            - layer 1: 2d convolution
                        in channels: 1 as its grayscale
                        out channels: defined in utils.py
            - layer 2: 2d convolution
                        in channels: defined in utils.py
                        out channels: defined in utils.py
            - layer 3: 2d convolution
                        in channels: defined in utils.py
                        out channels: defined in utils.py
            - layer 4: 2d convolution
                        in channels: defined in utils.py
                        out channels: 16 (assuming upscale_factor = 4)
        """
        super(SuperResolution, self).__init__()

        self.conv1 = nn.Conv2d(channels, out_channels, 5, padding=2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv4 = nn.Conv2d(out_channels, (upscale_factor ** 2), 
                               3, padding=1)

        #efficient sub-pixel layer
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        
    def forward(self, x):
        """
        Forward pass using Activation function

        Attributes:
            - Activation function: Leaky_relu, provides the best results.

        Returns:
            - Output tensor
        """
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.pixel_shuffle(self.conv4(x), upscale_factor=4)

        return x

