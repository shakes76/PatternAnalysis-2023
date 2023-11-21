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
        super(getUpscaleBlock, self).__init__()
        self.conv_block = nn.Conv2d(feature_size, feature_size * (upscale_factor ** 2), 3, padding='same')
        self.shuffle_block = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv_block(x)
        return self.shuffle_block(x)

#used to upscale the image, conv's with feature size x 4
#(although not 4 its a variable, we do 2x2)
#to increase the feature size, and then shuffles to make
#it go from deep to shallow but bigger

"""
Creates a set of feature extractors

Inputs:
feature_size            -           Number of input/output channels
num_convs               -           Number of cascaded convolutions
"""
class getConvBlock(nn.Module):
    def __init__(self, feature_size, num_convs):
        super(getConvBlock, self).__init__()
        conv_array = []
        for _ in range(num_convs):
            conv_array.append(nn.Conv2d(feature_size, feature_size, 3, padding='same'))
            # conv_array.append(nn.BatchNorm2d(feature_size))
            # conv_array.append(nn.Dropout2d())
            conv_array.append(nn.ReLU())
        self.conv_array = nn.Sequential(*conv_array)

    def forward(self, x):
        return self.conv_array(x)

#a conv with 32 -> 32 filters
#uses relu for performance
#we use for loop to add conv+relu to a list which has what we want to do
#to itself a bunch of times (conv->relu->conv etc)
#sequential makes it do the list items to itself in order

"""
Defines feature extraction and upscaling blocks to achieve a desired
upscaling factor.

Inputs:
upscale_factor          -           Desired upscaling factor
channels                -           Number of input channels 
feature_size            -           Desired number of features
num_convs               -           Number of cascaded convolutions in conv blocks
"""
class pixelCNN(nn.Module):
    def __init__(self, upscale_factor=4, channels=1, feature_size=32, num_convs=3):
        super(pixelCNN, self).__init__()

        # Define first convolution to create more channels
        self.conv_1 = nn.Conv2d(channels, feature_size, 3, padding='same')

        # Upscale in two-by-twos
        num_upsamplers = int(np.log2(upscale_factor))

        # Define array for conv blocks
        conv_and_upscale_blocks = []

        # Define the upscalers
        for _ in range(num_upsamplers):
            conv_and_upscale_blocks.append(getConvBlock(feature_size, num_convs))
            conv_and_upscale_blocks.append(getUpscaleBlock(feature_size, 2))
        # Define NN modules
        self.conv_and_upscale_blocks = nn.Sequential(*conv_and_upscale_blocks)

        # Define the final upsampling block to bring back to input channel
        self.conv_last = nn.Conv2d(feature_size, channels, 3, padding='same')

    def forward(self, x):
        # Increase feature size from 1 -> feature_size
        x = self.conv_1(x)

        # Perform feature extraction and upscaling x2
        x = self.conv_and_upscale_blocks(x)

        # Decrease feature size from feature_size -> 1
        return self.conv_last(x)
        
#uses the upscale/conv blocks we made 
#creates array for them

#run it (the block creator) through a for loop 2 times 
# (use log2 to find number of times needed 
# from upscale factor)
#one final conv to bring back to original channel size (1)
#use modulelist so the blocks can be sent to device

#if last was to bring it back to original, then forward
#is used to run through the blocks
#forward (1->32) then block (32->32) then last (32->1)

