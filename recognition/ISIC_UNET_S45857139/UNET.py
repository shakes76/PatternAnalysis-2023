import torch.nn as nn


def path(defaultChannels, initialChannels, maxChannels):
    """Completes the convolutional layer workflow of the specified channels

    Parameters:
        defaultChannels (int): the number of channels in the original input
        initialChannels (int): the initial resolution of the input at the beginning of encode/decode
        maxChannels (int): the max resolution of the input after the encode/decode

    Returns:
        (null)
    """
    channels = initialChannels

    # For encoding path (starting res < ending res)
    if initialChannels < maxChannels:
        nn.ReLU(nn.Conv2d(in_channels=defaultChannels, out_channels = initialChannels, kernel_size=3,stride=2))
        nn.ReLU(nn.Conv2d(in_channels=initialChannels, out_channels = initialChannels, kernel_size=3,stride=2))
        while channels <= maxChannels:
            nn.ReLU(nn.Conv2d(in_channels=channels, out_channels = channels*2, kernel_size=3,stride=2))
            channels *= 2
            nn.ReLU(nn.Conv2d(in_channels=channels, out_channels = channels, kernel_size=3,stride=2))
            if channels != maxChannels:
                nn.MaxPool2d(kernel_size=3,stride=2)

    # For decoding path (ending res > starting res)
    else:
        while channels >= maxChannels:
            nn.ReLU(nn.Conv2d(in_channels=channels, out_channels = channels/2, kernel_size=3,stride=2))
            channels /= 2
            nn.ReLU(nn.Conv2d(in_channels=channels, out_channels = channels, kernel_size=3,stride=2))

            if channels != maxChannels:
                nn.ReLU(nn.ConvTranspose2d(in_channels=1024,out_channels=512,kernel_size=3,stride=2))


class UNET(nn.Module):
    """UNet model for image segmentation."""
    def __init__(self):
        """Initializing the channels, and the workflow for encoding and decoding."""
        super(UNET, self).__init__()
        
        self.defaultChannels = 3
        self.initialChannels = 64
        self.maxChannels = 128
        self.endingChannels = 2

        self.encodePath = nn.Sequential(
            path(self.defaultChannels, self.initialChannels, self.maxChannels)
        )

        self.decodePath = nn.Sequential(
            path(self.defaultChannels, self.maxChannels, self.initialChannels)
        )

        def forward(self, x):
            """Passes the data through the encode and decode paths.

            Parameters: 
                x (tensor): the data as a tensor.

            Returns: 
                output (tensor): the output tensor image.

            """
            encode = self.encodePath(x)
            decode = self.decodePath(x)
            output = nn.ReLU(nn.Conv2d(self.initialChannels, self.endingChannels, kernel_size=3,stride=2))
            return output
        

        

        




