"""
A CNN model based on the Improved UNet architecture, with associated modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ContextModule(nn.Module):
    """
    This is the context module from the improved UNet architecture.
    "Each context module is in fact a pre-activation residual block with two
    3x3 convolutional layers and a dropout layer (pdrop = 0.3) in between."
    """
    def __init__(self, in_channels, out_channels):
        """
        Initialize the ContextModule.
        """
        super(ContextModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.relu1 = nn.LeakyReLU(negative_slope=1e-2)
        self.dropout = nn.Dropout2d(p=0.3)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.relu2 = nn.LeakyReLU(negative_slope=1e-2)
        
    def forward(self, x):
        """
        Forward pass of the ContextModule.
        """
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        out += x # residual connection
        return out

class LocalisationModule(nn.Module):
    """
    This is the localization module from the improved UNet architecture.
    A localization module consists of a 3x3 convolution followed by a 1x1 convolution that 
    halves the number of feature maps.
    """
    def __init__(self, in_channels, out_channels):
        """
        Initialize the LocalisationModule.
        """
        super(LocalisationModule, self).__init__()
        # self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass of the LocalisationModule.
        """
        out = self.conv1(x)
        out = self.conv2(out)
        return out


class UpsamplingModule(nn.Module):
    """
    An upsampling module consists of an upsampling layer that repeats the feature pixels 
    twice in each spatial dimension followed by a 3x3 convolution.
    """
    def __init__(self, in_channels, out_channels):
        """
        Initialize the UpsamplingModule.
        """
        super(UpsamplingModule, self).__init__()
        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        
    def forward(self, x):
        """
        Forward pass of the UpsamplingModule.
        """
        out = self.layers(x)
        return out


class ImprovedUNet(nn.Module):
    """
    This is the improved UNet model, it consists of the context aggregation pathway (encoder)
    and the localization pathway (decoder). The model is designed to outpreform the original
    UNet for medical image segmentation tasks.
    See https://arxiv.org/pdf/1802.10508v1.pdf for network architecture.
    """
    def __init__(self, in_channels=3, out_channels=1, features=[16,32,64,128,256]):
        """
        Initialize the ImprovedUNet model by creating all necessary layers.

        Args:
            in_channels (int, optional): Number of input channels. Defaults to 3 for RGB color images.
            out_channels (int, optional): Number of output channels. Defaults to 1 a for greyscale binary 
            segmentation mask.
            features (list, optional): The numbers of feature maps to extract (must be length 5 and each 
            entry must be double the previous entry). Defaults to [16,32,64,128,256].
        """
        super(ImprovedUNet, self).__init__()
        self.encoder_block1 = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=1e-2),
            ContextModule(features[0], features[0]), # 3 channels in, 16 channels out
        )
        self.encoder_block2 = nn.Sequential(
            nn.Conv2d(features[0], features[1], kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=1e-2),
            ContextModule(features[1], features[1]), # 16 channels in, 32 channels out
        )
        self.encoder_block3 = nn.Sequential(
            nn.Conv2d(features[1], features[2], kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=1e-2),
            ContextModule(features[2], features[2]), # 32 channels in, 64 channels out
        )
        self.encoder_block4 = nn.Sequential(
            nn.Conv2d(features[2], features[3], kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=1e-2),
            ContextModule(features[3], features[3]), # 64 channels in, 128 channels out
        )
        self.encoder_block5 = nn.Sequential(
            nn.Conv2d(features[3], features[4], kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=1e-2),
            ContextModule(features[4], features[4]), # 128 channels in, 256 channels out
            # upsampling module in last encoder block increases spatial dimensions by 2 for decoder
            UpsamplingModule(features[4], features[3]), # 256 channels in, 128 channels out
        )
        # Upsampling modules half the number of feature maps but after upsampling, the output
        # is concatenated with the skip connection, so the number of feature maps is doubled
        # the localization modules then halve the number of feature maps again
        self.decoder_block1 = nn.Sequential(
            LocalisationModule(features[4], features[3]), # 256 channels in, 128 channels out
            UpsamplingModule(features[3], features[2]), # 128 channels in, 64 channels out
        )
        # these decoder layers need to be split up to allow for skip connections
        self.localisation2 = LocalisationModule(features[3], features[2]) # 128 channels in, 64 channels out
        self.up3 = UpsamplingModule(features[2], features[1]) # 64 channels in, 32 channels out, double spatial dimensions
        self.localisation3 = LocalisationModule(features[2], features[1]) # 64 channels in, 32 channels out
        self.up4 = UpsamplingModule(features[1], features[0]) # 32 channels in, 16 channels out, double spatial dimensions
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[1], features[1], kernel_size=1), # 32 channels in, 32 channels out, final convolutional layer
            nn.LeakyReLU(negative_slope=1e-2),
            nn.Conv2d(features[1], out_channels, kernel_size=1) # 32 channels in, 1 channel out, segmentation layer
        )
        
        self.segmentation1 = nn.Conv2d(features[2], out_channels, kernel_size=1) # 64 channels in, 1 channels out, segmentation layer
        self.upscale1 = nn.Upsample(scale_factor=2, mode='nearest') # upscale the segmentation layer to match the dimensions of the output
        self.segmentation2 = nn.Conv2d(features[1], out_channels, kernel_size=1) # 32 channels in, 1 channels out, segmentation layer
        self.upscale2 = nn.Upsample(scale_factor=2, mode='nearest') # upscale the segmentation layer to match the dimensions of the output
        
    
    def forward(self, x):
        """Forward pass of the ImprovedUNet model to generate a binary segmentation mask.

        Args:
            x (torch.Tensor): An input image in tensor form.

        Returns:
            torch.Tensor: Binary segmentation mask in tensor form
        """
        skip_connections = []
        
        x = self.encoder_block1(x) # 3 channels in, 16 channels out
        skip_connections.append(x)
        x = self.encoder_block2(x) # 16 channels in, 32 channels out
        skip_connections.append(x)
        x = self.encoder_block3(x) # 32 channels in, 64 channels out
        skip_connections.append(x)
        x = self.encoder_block4(x) # 64 channels in, 128 channels out
        skip_connections.append(x)
        # bottleneck layer
        x = self.encoder_block5(x) # 128 channels in, 128 channels out
        
        # use skip connections as a stack to allow for easy popping
        # concatenate the skip connection with 128 channels with the bottleneck layer with 128 channels
        x = torch.cat((x, skip_connections.pop()), dim=1)
        
        x = self.decoder_block1(x) # 256 channels in, 64 channels out
        # concatenate the skip connection with 64 channels with the bottleneck layer with 64 channels
        x = torch.cat((x, skip_connections.pop()), dim=1)
        
        x = self.localisation2(x) # 128 channels in, 64 channels out
        # additional skip connections for segmentation layers in the decoder
        seg_connection1 = self.segmentation1(x) # 64 channels in, 1 channel out
        seg_connection1 = self.upscale1(seg_connection1)
        
        x = self.up3(x)
        # concatenate the skip connection with 32 channels with the bottleneck layer with 32 channels
        x = torch.cat((x, skip_connections.pop()), dim=1)
        
        x = self.localisation3(x) # 64 channels in, 32 channels out
        # additional skip connections for segmentation layers in the decoder
        seg_connection2 = self.segmentation2(x) # 32 channels in, 1 channel out
        # element wise addition of the segmentation connections
        seg_connection2 += seg_connection1
        seg_connection2 = self.upscale2(seg_connection2) # upscale the segmentation connection to match the dimensions of the output
        
        x = self.up4(x)
        # concatenate the skip connection with 16 channels with the bottleneck layer with 16 channels
        x = torch.cat((x, skip_connections.pop()), dim=1)
        
        x = self.final_conv(x) # 32 channels in, 1 channel out, includes final convolutional layer and segmentation layer
        
        # element wise addition of the segmentation connections
        x += seg_connection2
        
        return x # no final activation function
