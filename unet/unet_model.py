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