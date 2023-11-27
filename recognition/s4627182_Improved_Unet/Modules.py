""" Full assembly of the parts to form the complete network """
from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch


class ContextModule(nn.Module):
    """
    Context Module: Consists of two convolutional layers for feature extraction and a dropout layer
    for regularization, aimed at capturing and preserving the context information in the features.
    """

    def __init__(self, in_channels, out_channels):
        """
        Initialize the Context Module.

        Parameters:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
        """

        super(ContextModule, self).__init__()
        # 2 3x3 convolution layer followed by instance normalization and leaky ReLU activation
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout2d(p=0.3)

        def forward(self, x):
            """
            Forward pass through the context module. Input is put through 2 3x3 stride 1 convolutions with a dropout
            layer in between

            Parameters:
            - x (Tensor): The input tensor.

            Returns:
            - Tensor: The output tensor after passing through the context module.
            """
            x = self.conv1(x)
            x = self.dropout(x)
            x = self.conv2(x)
            return x


class SegmentationLayer(nn.Module):
    """
    SegmentationLayer: A convolutional layer specifically utilized to generate a segmentation map.
    """

    def __init__(self, in_channels, out_channels):
        """
        Initialize the SegmentationLayer.

        Parameters:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels, often equal to the number of classes in segmentation.
        """
        super(SegmentationLayer, self).__init__()
        # A convolutional layer that produces segmentation map
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        Forward pass through the SegmentationLayer.

        Parameters:
        - x (Tensor): The input tensor.

        Returns:
        - Tensor: The output tensor after applying the convolution, serving as a segmentation map.
        """
        # Applying convolution
        x = self.conv(x)
        return x


class UpscalingLayer(nn.Module):
    """
    UpscalingLayer: A layer designed to upscale feature maps by a factor of 2.
    """

    def __init__(self, scale_factor=2, mode='nearest'):
        """
        Initialize the UpscalingLayer.

        Parameters:
        - scale_factor (int, optional): Factor by which to upscale the input. Default is 2.
        - mode (str, optional): Algorithm used for upscaling: 'nearest', 'bilinear', etc. Default is 'nearest'.
        """
        super(UpscalingLayer, self).__init__()
        # An upsampling layer that increases the spatial dimensions of the feature map
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)

    def forward(self, x):
        """
        Forward pass through the UpscalingLayer.

        Parameters:
        - x (Tensor): The input tensor.

        Returns:
        - Tensor: The output tensor after applying the upscaling, having increased spatial dimensions.
        """
        # Applying upscaling
        x = self.upsample(x)
        return x


class LocalisationModule(nn.Module):
    """
    Localisation Module: Focused on up-sampling the received feature map and reducing the
    number of feature channels, working towards recovering the spatial resolution of the input data.
    """

    def __init__(self, in_channels, out_channels):
        """
        Initialize the Localisation Module.

        Parameters:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
        """
        super(LocalisationModule, self).__init__()
        # Using a simple upscale by repeating the feature pixels twice
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # 3x3 convolution to process concatenated features
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 1x1 convolution to reduce the number of feature maps
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        Forward pass through the localisation module. Input is put through 2 3x3 stride 1 convolutions
        with leaky ReLU applied in between

        Parameters:
        - x (Tensor): The input tensor.

        Returns:
        - Tensor: The output tensor after passing through the localisation module.
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        return x


class UpsamplingModule(nn.Module):
    """
    Upsampling Module: Handles the up-sampling of feature maps in the decoder part of the UNet,
    contributing to incrementing the spatial dimensions of the input feature map.
    """

    def __init__(self, in_channels, out_channels):
        """
        Initialize the Upsampling Module.

        Parameters:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
        """
        super(UpsamplingModule, self).__init__()
        # Using a simple upscale by repeating the feature pixels twice
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # 3x3 convolution that halves the number of feature maps
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        Forward pass through the upsampling module. First the input is upsampled, then undergoes stride 1
        3x3 convolution followed by leaky ReLU.

        Parameters:
        - x (Tensor): The input tensor.

        Returns:
        - Tensor: The output tensor after passing through the upsampling module.
        """
        x = self.upsample(x)
        x = self.conv(x)
        x = F.relu(x)

        return x


class UNet_For_Brain(nn.Module):
    """
    UNet2D: An Improved U-Net model implmented as the provided Improved U-Net paper
    """

    def __init__(self, in_channels, num_classes=2):
        """
        Initialize the UNet2D model.

        Parameters:
        - in_channels (int): Number of input channels.
        - num_classes (int): Number of output classes for segmentation.
        """
        super(UNet_For_Brain, self).__init__()

        # Encoder
        self.enc1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.context1 = ContextModule(16, 16)
        self.enc2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.context2 = ContextModule(32, 32)
        self.enc3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.context3 = ContextModule(64, 64)
        self.enc4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.context4 = ContextModule(128, 128)

        # Bottleneck
        self.bottleneck = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bottleneck_context = ContextModule(256, 256)
        self.up_bottleneck = UpsamplingModule(256, 128)

        # Decoder
        self.local1 = LocalisationModule(256, 128)
        self.up1 = UpsamplingModule(128, 64)

        self.local2 = LocalisationModule(128, 64)
        self.up2 = UpsamplingModule(64, 32)

        self.seg1 = SegmentationLayer(64, num_classes)
        self.upsample_seg1 = UpscalingLayer()

        self.local3 = LocalisationModule(64, 32)
        self.up3 = UpsamplingModule(32, 16)

        self.seg2 = SegmentationLayer(32, num_classes)
        self.upsample_seg2 = UpscalingLayer()

        self.final_conv = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.seg3 = SegmentationLayer(32, num_classes)
        self.upsample_seg3 = UpscalingLayer()

    def forward(self, x):
        """
        Define the forward pass through the UNet2D model.

        Parameters:
        - x (Tensor): Input tensor.

        Returns:
        - Tensor: The output tensor after passing through the model.
        """
        y1 = self.enc1(x)
        x1 = self.context1(y1)
        x1 = x1 + y1

        y2 = self.enc2(x1)
        x2 = self.context2(y2)
        x2 = x2 + y2

        y3 = self.enc3(x2)
        x3 = self.context3(y3)
        x3 = x3 + y3

        y4 = self.enc4(x3)
        x4 = self.context4(y4)
        x4 = x4 + y4

        # Bottleneck
        bottleneck_conv = self.bottleneck(x4)

        bottleneck = self.bottleneck_context(bottleneck_conv)
        bottleneck = bottleneck + bottleneck_conv

        up_bottleneck = self.up_bottleneck(bottleneck)

        # Decoder
        x = self.local1(torch.cat((x4, up_bottleneck), dim=1))
        x = self.up1(x)

        x = self.local2(torch.cat((x3, x), dim=1))
        seg1 = self.seg1(x)
        x = self.up2(x)

        seg1_upsampled = self.upsample_seg1(seg1)

        x = self.local3(torch.cat((x2, x), dim=1))
        seg2 = self.seg2(x)
        x = self.up3(x)

        seg12 = seg1_upsampled + seg2
        seg12_up = self.upsample_seg2(seg12)

        x = self.final_conv(torch.cat((x1, x), dim=1))

        seg3 = self.seg3(x)
        seg123 = seg3 + seg12_up

        out = seg123
        # out = nn.functional.softmax(seg123, dim=1)
        # out = torch.sigmoid(seg123)
        # print("out shape: ", out.size())

        return out

# Updated
