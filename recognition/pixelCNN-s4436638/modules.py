import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

"""
Creates an upsampling block by increasing the feature size by a factor of
upscale_factor ** 2, then PixelShuffles to increase the image size.

Inputs:
feature_size            -           Number of input/output channels
upscale_factor          -           The upscale factor of the image
"""
class getUpscaleBlock(nn.Module):
    def __init__(self, feature_size, upscale_factor):
        super(getUpscaleBlock).__init__()
        self.conv_block = nn.Conv2d(feature_size, feature_size * (upscale_factor ** 2), 3, padding='same')
        self.shuffle_block = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv_block(x)
        return self.shuffle_block(x)
    
"""
Creates a set of feature extractors

Inputs:
feature_size            -           Number of input/output channels
num_convs               -           Number of cascaded convolutions
"""
class getConvBlock(nn.Module):
    def __init__(self, feature_size, num_convs):
        super(getConvBlock).__init__()
        conv_array = []
        for _ in range(num_convs):
            conv_array.append(nn.Conv2d(feature_size, feature_size, 3, padding='same'))

        self.conv_array = nn.Sequential(*conv_array)

    def forward(self, x):
        return self.conv_array(x)

"""
Defines feature extraction and upscaling blocks to achieve a desired
upscaling factor.

Inputs:
upscale_factor          -           Desired upscaling factor
channels                -           Number of input channels 
feature_size            -           Desired number of features
num_convs               -           Number of cascadedd convolutions in conv blocks
"""
class pixelCNN(nn.Module):
    def __init__(self, upscale_factor=4, channels=1, feature_size=32, num_convs=3):
        super(pixelCNN, self).__init__()

        # Define first convolution to create more channels
        self.conv_1 = nn.Conv2d(channels, feature_size, 3, padding='same')

        # Upscale in two-by-twos
        num_upsamplers = int(np.log2(upscale_factor))

        # Define array for conv blocks
        conv_blocks = []

        # Define array for upscale blocks
        upscale_blocks = []

        # Define the upscalers
        for _ in range(num_upsamplers):
            conv_blocks.append(getConvBlock(feature_size, num_convs))
            upscale_blocks.append(getUpscaleBlock(feature_size, 2))

        # Define the final upsampling block to bring back to input channel
        self.conv_last = nn.Conv2d(feature_size, channels, 3, padding='same')

        # Define NN modules
        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.upscale_blocks = nn.ModuleList(upscale_blocks)

    def forward(self, x):
        # Increase feature size
        x = self.conv_1(x)
        # Perform feature extraction and upscaling
        for i, conv in enumerate(self.conv_blocks):
            x = conv(x)
            x = self.upscale_blocks[i](x)
        # Decrease feature size
        return self.conv_last(x)
        

