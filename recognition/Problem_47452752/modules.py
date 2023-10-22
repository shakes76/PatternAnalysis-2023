"""
Contains the source code of the components in my model. Each component is implemented as a class or function. 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Context(nn.Module):
    """
    Encoder that computes the activations in the context pathway. This class behaves as the 'context module' from the paper.
    Each Context module is a pre-activation residual block with two 3x3x3 convolutional layers and a dropout layer (p = 0.3) in between.
    Instance normalization and leaky ReLU is used throughout the network.
    """

    def __init__(self, in_channels):
        super(Context, self).__init__()
        # We use leakyReLU with negative slope 1e-2
        self.relu = nn.LeakyReLU(negative_slope=1e-2, inplace=False)
        # 3x3x3 convolutional layer that preserves the input size
        self.conv = nn.Conv3d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        # Dropout layer with p_dropout = 0.3
        self.dropout = nn.Dropout(p=0.3)
        # Normalize the batch with instance normalization
        self.norm = nn.InstanceNorm3d(in_channels)

    def forward(self, x):
        shortcut = x

        x = self.relu(self.norm(x))
        x = self.dropout(x)
        x = self.relu(self.norm(x))

        return x + shortcut


class Upsample(nn.Module):
    """
    Upsampling module used to tranfer information from low resolution feature maps into high resolution fearure maps.
    We use a simple upscale that repeats the feature voxels twice in each spatial dimension, followed by a 3x3x3 convolution
    that halves the number of feature maps.
    """

    def __init__(self, in_channels):
        super(Upsample, self).__init__()
        self.upsample = nn.Sequential(
            # Upscale the feature voxels
            nn.Upsample(scale_factor=2, mode="nearest"),
            # Normalize the batch useing instance normalization
            nn.InstanceNorm3d(in_channels),
            # Introduce non-linearity using leakyReLU with negative slope 1e-2
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
            # Apply 3x3x3 convolution that havles number of feature maps
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, skip_channels):
        x = self.upsample(x)
        return torch.cat([x, skip_channels], dim=1)


class Localisation(nn.Module):
    def __init__(self, in_channels):
        super(Localisation, self).__init__()
        # We use leakyReLU with negative slope 1e-2
        self.relu = nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        # We use a 3x3x3 convolution first
        self.norm1 = nn.InstanceNorm3d(in_channels)
        self.conv1 = nn.Conv3d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        # We halve the number of feature maps by using a 1x1x1 convolution
        self.norm2 = nn.InstanceNorm3d(in_channels // 2)
        self.conv2 = nn.Conv3d(
            in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        x = self.conv1(self.relu(self.norm1(x)))
        x = self.conv2(self.relu(self.norm2(x)))
        return x


class Segmentation(nn.Module):
    pass


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(UNet, self).__init__()

        # Context modules (encoders)
        self.context1 = Context(16)
        self.context2 = Context(32)
        self.context3 = Context(64)
        self.context4 = Context(128)
        self.context5 = Context(256)

        # Upsampling modules
        self.up1 = Upsample(256)
        self.up2 = Upsample(128)
        self.up3 = Upsample(64)
        self.up4 = Upsample(32)

        # Localisation modules
        self.local1 = Localisation(128)
        self.local2 = Localisation(64)
        self.local3 = Localisation(32)

        # Convolutions that connect context modules, used for downsampling
        self.conv1 = nn.Conv3d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)

        # Final output layer
        self.out_conv = nn.Conv3d(16, out_channels, kernel_size=1)

        # To upscale the segmentaion layers 
        self.upscale = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        # Segmentation layers 
        self.segment1 = nn.Conv3d(64, num_classes, kernel_size=1, stride=1)
        self.segment2 = nn.Conv3d(32, num_classes, kernel_size=1, stride=1)
        self.segment3 = nn.Conv3d(16, num_classes, kernel_size=1, stride=1)
        

    def forward(self, x):

        # Context Pathway
        c1 = self.context1(self.conv1(x))
        c2 = self.context2(self.conv2(c1))
        c3 = self.context3(self.conv3(c2))
        c4 = self.context4(self.conv4(c3))
        x = self.context5(self.conv5(c4))

        # Decoding Pathway (upsample + localisation)
        l1 = self.local1(self.up1(x, c4))
        l2 = self.local2(self.up2(l1, c3)) # TODO
        l3 = self.local3(self.up3(l2, c2)) # TODO
        x = self.out_conv(self.up4(l3, c1))

        # Deep Supervision 
        s1 = self.segment1() 



        # Final output
        return self.out_conv(dec2)
