"""
Contains the source code of the components in my model. Each component is implemented as a class or function. 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Encoder that computes the activations in the context pathway. This class behaves as the 'context module' from the paper.
    Each Encoder module is a pre-activation residual block with two 3x3x3 convolutional layers and a dropout layer (p = 0.3) in between.
    Instance normalization and leaky ReLU is used throughout the network.
    """

    def __init__(self, in_channels):
        # We use leakyReLU with negative slope 1e-2
        self.relu = nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        # 3x3x3 convolutional layer that preserves the input size
        self.conv = nn.Conv3d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        # Dropout layer with p_dropout = 0.3
        self.dropout = nn.Dropout(p=0.3)
        # Normalize the batch with instance normalization
        self.norm = nn.InstanceNorm3d()

    def forward(self, x):
        shortcut = x

        residual = self.relu(self.norm(x))
        residual = self.dropout(residual)
        residual = self.relu(self.norm(x))

        return residual + shortcut


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        pass

    def forward(self, x, skip_connection):
        # halve the in_features, concativate with skip features
        pass


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Context Pathway (encoders)
        self.enc1 = Encoder(16)
        self.enc2 = Encoder(32)
        self.enc3 = Encoder(64)
        self.enc4 = Encoder(128)
        self.enc5 = Encoder(256)

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
