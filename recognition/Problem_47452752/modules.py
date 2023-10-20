"""
Contains the source code of the components in my model. Each component is implemented as a class or function. 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        # leakyReLU, convolution, normalization
        pass

    def forward(self, x):
        # element-wise summation of pre and post processed features
        pass


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        pass

    def forward(self, x, skip_connection):
        # halve the in_features, concativate with skip features, then halve again
        pass


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Context Pathway (encoders)
        self.enc1 = Encoder(in_channels, 16)
        self.enc2 = Encoder(32, 32)
        self.enc3 = Encoder(64, 64)
        self.enc4 = Encoder(128, 128)
        self.enc5 = Encoder(256, 256)

        # Convolutions for downsampling
        self.down1 = nn.Conv3d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.down2 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)
        self.down3 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.down4 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.down5 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)

        # Localization Pathway
        self.up1 = Decoder(256, 64)
        self.up2 = Decoder(64, 64)
        self.up3 = Decoder(128, 128)

        # Final output layer
        self.out_conv = nn.Conv3d(16, out_channels, kernel_size=1)

    def forward(self, x):
        # Context Pathway
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.down1(enc1))
        enc3 = self.enc3(self.down2(enc2))

        # Localization Pathway with skip connections (decoding)
        dec1 = self.up1(enc3, enc2)
        dec2 = self.up2(dec1, enc1)

        # Final output
        return self.out_conv(dec2)
