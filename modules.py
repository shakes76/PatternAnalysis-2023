import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)
    
class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels, out_channels, num_levels=4):
        super(UNetPlusPlus, self).__init__()

        self.num_levels = num_levels

        # Define the encoder and decoder paths
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for level in range(num_levels):
            in_channels = in_channels if level == 0 else out_channels
            out_channels *= 2

            # Encoder block with convolution layers
            encoder = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            self.encoders.append(encoder)

            # Decoder block with upconvolution layers
            decoder = nn.Sequential(
                nn.ConvTranspose2d(out_channels, out_channels // 2, kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            self.decoders.append(decoder)

        # Final output convolution
        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        features = []  # Store skip connections

        # Encoding path
        for level in range(self.num_levels):
            x = self.encoders[level](x)
            if level < self.num_levels - 1:
                features.append(x)  # Store skip connection

        # Decoding path
        for level in range(self.num_levels - 1, 0, -1):
            x = self.decoders[level - 1](x)
            x = torch.cat((x, features[level - 1]), dim=1)  # Concatenate skip connection
            if level > 1:
                x = self.decoders[level - 1](x)

        # Final output
        x = self.final_conv(x)
        return x

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predicted, target):
        intersection = (predicted * target).sum()
        union = predicted.sum() + target.sum()
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice  # We often minimize the loss, so return 1 - dice
