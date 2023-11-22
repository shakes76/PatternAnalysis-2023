import torch.nn as nn
import torch.nn.functional as F


class SuperResolution(nn.Module):
    """
    A model for image super resolution using an
    Efficient Sub-Pixel convolutional neural network.

    This model takes low resolution images (60x64) and upscales
    them by a factor of 4 (to 240x256) using four convolutional
    layers and a pixel shuffle operation.
    """
    def __init__(self):
        """
        Initialises the model and builds the required layers within
        it.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(1, 400, 7, 1, 3)
        self.conv2 = nn.Conv2d(400, 100, 5, 1, 2)
        self.conv3 = nn.Conv2d(100, 50, 5, 1, 2)
        self.conv4 = nn.Conv2d(50, 4 ** 2, 3, 1, 1)
        self.pixel = nn.PixelShuffle(4)
        
    
    def forward(self, x):
        """
        Completes one forward step of the model from the input x.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pixel(self.conv4(x))
        return x