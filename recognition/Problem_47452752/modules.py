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
        self.relu = nn.LeakyReLU(negative_slope=1e-2, inplace=True)
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


class Up(nn.Module):
    """
    Upsampling module used to tranfer information from low resolution feature maps into high resolution fearure maps.
    We use a simple upscale that repeats the feature voxels twice in each spatial dimension, followed by a 3x3x3 convolution
    that halves the number of feature maps.
    """

    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.upsample = nn.Sequential(
            # Upscale the feature voxels
            nn.Upsample(scale_factor=2, mode="nearest"),
            # Normalize the batch useing instance normalization
            nn.InstanceNorm3d(in_channels),
            # Introduce non-linearity using leakyReLU with negative slope 1e-2
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
            # Apply 3x3x3 convolution that havles number of feature maps
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, skip_channels):
        x = self.conv(self.relu(self.norm(x)))
        return torch.cat([x, skip_channels], dim=1)


class Localisation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Localisation, self).__init__()
        # We use leakyReLU with negative slope 1e-2
        self.relu = nn.LeakyReLU(negative_slope=1e-2, inplace=True)

        # The Localisation module involves two rounds of normalizations and convoltions

        # Round 1, we apply a 3x3x3 convolution
        self.norm1 = nn.InstanceNorm3d(in_channels)
        self.conv1 = nn.Conv3d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        # Round 2, we halve the number of feature maps using a 1x1x1 convolution
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.conv2 = nn.Conv3d(
            in_channels // 2, out_channels, kernel_size=3, stride=1, padding=1
        )

        def forward(self, x):
            x = self.conv1(self.relu(self.norm1(x)))
            x = self.conv2(self.relu(self.norm2(x)))
            return x

    def forward(self, x, skip_connection):
        upsampled = self.upsample(x)
        upsampled = self.relu(self.norm(upsampled))
        upsampled = self.conv(upsampled)

        # Concatenate with skip features from the context pathway
        x = torch.cat([x, skip_features], dim=1)

        # halve the in_features, concativate with skip features
        pass


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Context Pathway (encoders)
        self.enc1 = Context(16)
        self.enc2 = Context(32)
        self.enc3 = Context(64)
        self.enc4 = Context(128)
        self.enc5 = Context(256)

        # Convolutions that connect context modules, used for downsampling
        self.down1 = nn.Conv3d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.down2 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)
        self.down3 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.down4 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.down5 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)

        # Localization Pathway
        self.up1 = Decoder(256, 128)
        self.up2 = Decoder(128, 64)
        self.up3 = Decoder(64, 32)
        self.up4 = Decoder(32, 16)

        # Final output layer
        self.out_conv = nn.Conv3d(16, out_channels, kernel_size=1)

    def forward(self, x):
        # Context Pathway
        enc1 = self.enc1(self.down1(x))
        enc2 = self.enc2(self.down2(enc1))
        enc3 = self.enc3(self.down3(enc2))
        enc4 = self.enc4(self.down1(enc3))
        enc5 = self.enc5(self.down2(enc4))

        # Localization Pathway with skip connections (decoding)
        dec1 = self.up1(enc3, enc2)
        dec2 = self.up2(dec1, enc1)

        # Final output
        return self.out_conv(dec2)
