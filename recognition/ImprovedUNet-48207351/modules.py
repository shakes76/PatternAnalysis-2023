import torch
print('__version__')
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
    
class ContextModule(nn.Module):
    """
    Context Module in the Improved UNET architecture.

    This module is responsible for processing features in the context pathway.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        pdrop (float, optional): Dropout probability. Default is 0.3.

    Attributes:
        conv1 (nn.Conv3d): 3D convolutional layer 1.
        conv2 (nn.Conv3d): 3D convolutional layer 2.
        dropout (nn.Dropout3d): 3D dropout layer.
        norm (nn.LazyInstanceNorm3d): Lazy 3D instance normalization layer.
        relu (nn.LeakyReLU): Leaky ReLU activation function.

    Methods:
        forward(x):
            Forward pass through the context module.
    """
    def __init__(self, in_channels, out_channels, pdrop=0.3):
            super(ContextModule, self).__init__()
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
            self.conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
            self.dropout = nn.Dropout3d(pdrop)
            self.norm = nn.LazyInstanceNorm3d(out_channels)
            self.relu = nn.LeakyReLU(0.01, inplace=True)


    def forward(self, x):
        out = self.norm(self.relu(self.conv1(x)))
        out = self.norm(self.relu(self.conv2(out)))
        out = self.dropout(out)
        return out
    

class LocalizationModule(nn.Module):
    """
    Localization Module in the Improved UNET architecture.

    This module is responsible for processing features in the localization pathway.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.

    Attributes:
        upsample (nn.ConvTranspose3d): 3D transposed convolutional layer for upsampling.
        conv1 (nn.Conv3d): 3D convolutional layer 1.
        conv2 (nn.Conv3d): 3D convolutional layer 2.
        norm (nn.InstanceNorm3d): 3D instance normalization layer.
        relu (nn.LeakyReLU): Leaky ReLU activation function.

    Methods:
        forward(x, context_features):
            Forward pass through the localization module.

    """
    def __init__(self, in_channels, out_channels):
        super(LocalizationModule, self).__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv1 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=1)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.relu = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x, context_features):
        x = self.norm(self.relu(self.upsample(x)))
        # Concatenate with features from the context pathway
        x = torch.cat((x, context_features), dim=1)
        x = self.norm(self.relu(self.conv1(x)))
        x = self.norm(self.relu(self.conv2(x)))
        return x

class ImprovedUNET(nn.Module):
    """
    Improved U-Net Architecture.

    This model consists of a context pathway and a localization pathway.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        levels (int, optional): Number of levels in the U-Net architecture. Default is 4.
        features (int, optional): Number of features in each level. Default is 4.

    Attributes:
        context_ml (nn.ModuleList): List of ContextModules in the context pathway.
        localization_ml (nn.ModuleList): List of LocalizationModules in the localization pathway.
        supervision_layers (nn.ModuleList): List of supervision layers for each level.

    Methods:
        forward(x):
            Forward pass through the Improved U-Net.
    """
    def __init__(self, in_channels, out_channels, levels=4, features=4):

        super(ImprovedUNET, self).__init__()
        self.context_ml = nn.ModuleList()
        self.localization_ml = nn.ModuleList()
        self.supervision_layers = nn.ModuleList()

        # Context pathway of improved UNET
        for level in range(levels):
            in_channels = in_channels if level == 0 else features * 2
            out_channels = features * 2
            self.context_ml.append(ContextModule(in_channels, out_channels))

        # Localization pathway of improved UNET
        for level in range(levels - 1, 0, -1):
            in_channels = features * 2
            out_channels = features
            self.localization_ml.append(LocalizationModule(in_channels, out_channels))

        # Create supervision layers
        for level in range(levels):
            out = nn.Conv3d(features, out_channels, kernel_size=1)
            self.supervision_layers.append(out)

    def forward(self, x):
        context_features = []

        # Forward pass through the context pathway
        for context_module in self.context_ml:
            x = context_module(x)
            context_features.append(x)

        # Initialize the output
        output = self.supervision_layers[0](context_features[-1])

        # Forward pass through the localization pathway
        for level, localization_module in enumerate(self.localization_ml):
            x = localization_module(x, context_features[-level - 2])
            if level < len(self.supervision_layers):
                output += self.supervision_layers[level](x)

        return output
    