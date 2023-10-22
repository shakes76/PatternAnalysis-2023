""" Full assembly of the parts to form the complete network """

from .unet_parts import *



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

