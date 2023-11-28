"""
File: modules.py
Author: Maia Josang
Description: Contains the Super-Resolution model implementation.
"""

import torch
import torch.nn as nn

# Define the upscale factor for super-resolution
upscale_factor = 4

class SuperResolutionModel(nn.Module):
    def __init__(self, upscale_factor=4, channels=3):
            super(SuperResolutionModel, self).__init__()
            self.conv_args = {
                "kernel_size": 3,
                "padding": 1,
            }
            # Input convolutional layer: Converts input channels to 64 feature maps
            self.inputs = nn.Conv2d(channels, 64, kernel_size=5, padding=2)
            # Convolutional layers for feature extraction
            self.conv1 = nn.Conv2d(64, 64, **self.conv_args)
            self.conv2 = nn.Conv2d(64, 32, **self.conv_args)
            # Convolutional layer for upscaling, output channels depend on the upscale factor
            self.conv3 = nn.Conv2d(32, channels * (upscale_factor ** 2), **self.conv_args)

    def forward(self, out):
        # Apply ReLU activation 
        out = torch.relu(self.inputs(out))
        out = torch.relu(self.conv1(out))
        out = torch.relu(self.conv2(out))
        # Apply the final convolutional layer without activation
        out = self.conv3(out)
        # Pixel shuffle layer for upscaling the image
        out = torch.nn.functional.pixel_shuffle(out, upscale_factor)
        return out
