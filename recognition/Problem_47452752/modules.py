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
        # 3x3 convolutional layer that preserves the input size
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        # Dropout layer with p_drop = 0.3
        self.dropout = nn.Dropout(p=0.3)
        # Instance normalization of the input is used
        self.norm = nn.InstanceNorm2d(in_channels)

    def forward(self, x):
        # Keep track of the initial input
        shortcut = x
        # First convolutional layer
        x = F.leaky_relu(self.norm(x), negative_slope=1e-2)
        x = self.conv(x)
        # Dropout layer
        x = self.dropout(x)
        # Second convolution layer
        x = F.leaky_relu(self.norm(x), negative_slope=1e-2)
        x = self.conv(x)
        # Return the residual output
        return x + shortcut


class Upsampling(nn.Module):
    """
    Upsampling module used to tranfer information from low resolution feature maps into high resolution fearure maps.
    We use a simple upscale that repeats the feature voxels twice in each spatial dimension, followed by a 3x3x3 convolution
    that halves the number of feature maps. Instance normalization and leaky ReLU is used throughout the network.
    """

    def __init__(self, in_channels):
        super(Upsampling, self).__init__()
        # Upsamping components:
        self.up_norm = nn.InstanceNorm2d(in_channels)
        self.up_conv = nn.Conv2d(
            in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1
        )

        # Localisation components:
        self.merged_norm = nn.InstanceNorm2d(in_channels)
        self.merged_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.half_norm = nn.InstanceNorm2d(in_channels // 2)
        self.half_conv = nn.Conv2d(
            in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x, skip_features, local: bool = True):
        """
        Forward pass of the Upsampling module, with an optional localisation step.

        Parameters:
        - x (torch.Tensor): Input tensor, representing the activations from a deeper layer.
        - skip_features (torch.Tensor): Activations from the corresponding encoder layer, for the skip connection.
        - local (bool, optional): If True, the localisation operations are applied after upsampling. Default is True.

        Returns:
        - torch.Tensor: Upsampled (and optionally localised) feature map.

        The function first upsamples the input tensor 'x' and then concatenates the result with the 'skip_features'.
        If 'local' is True, it subsequently applies the localisation steps to refine the feature maps further.
        """
        # Upsampling Module
        upsampled = F.interpolate(x, scale_factor=2, mode="nearest")
        upsampled = F.leaky_relu(self.up_norm(upsampled), negative_slope=1e-2)
        upsampled = self.up_conv(upsampled)

        # Concatenate upsampled features with context features
        merged = torch.cat([upsampled, skip_features], dim=1)

        if local is False:
            return merged

        # Localisation Module
        localised = F.leaky_relu(self.merged_norm(merged), negative_slope=1e-2)
        localised = self.merged_conv(localised)  # First convolutional layer

        localised = F.leaky_relu(self.half_norm(localised), negative_slope=1e-2)
        localised = self.half_conv(localised)  # Second convolutional layer

        return localised


class Segmentation(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Segmentation, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1, stride=1)

    def forward(self, x, other_layer, upscale=True):
        x = self.conv(x)
        if other_layer is not None:
            x += other_layer
        if not upscale:
            return x
        return F.interpolate(x, scale_factor=2, mode="nearest")


class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UNet, self).__init__()

        # Context modules
        self.context1 = Context(16)
        self.context2 = Context(32)
        self.context3 = Context(64)
        self.context4 = Context(128)
        self.context5 = Context(256)

        # Upsampling modules
        self.up1 = Upsampling(256)
        self.up2 = Upsampling(128)
        self.up3 = Upsampling(64)
        self.up4 = Upsampling(32)

        # Convolutional layer used on input channel
        self.input_conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)

        # Convolutional layers that connect context modules
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        # Convolutional layer for the localisation pathway output
        self.end_conv = nn.Conv2d(32, 32, kernel_size=1)

        # Segmentation layers
        self.segment1 = Segmentation(64, num_classes)
        self.segment2 = Segmentation(32, num_classes)
        self.segment3 = Segmentation(32, num_classes)

    def forward(self, x):
        # Context Pathway
        cntx_1 = self.context1(self.input_conv(x))
        cntx_2 = self.context2(self.conv1(cntx_1))
        cntx_3 = self.context3(self.conv2(cntx_2))
        cntx_4 = self.context4(self.conv3(cntx_3))
        cntx_5 = self.context5(self.conv4(cntx_4))

        # Localization Pathway
        local_1 = self.up1(cntx_5, cntx_4)
        local_2 = self.up2(local_1, cntx_3)
        local_3 = self.up3(local_2, cntx_2)
        local_out = self.end_conv(self.up4(local_3, cntx_1, False))

        # Deep Supervision
        seg_1 = self.segment1(local_2, None)
        seg_2 = self.segment2(local_3, seg_1)
        seg_3 = self.segment3(local_out, seg_2, upscale=False)

        # Apply softmax and return
        return F.softmax(seg_3, dim=1)

