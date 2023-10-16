"""
File containing the module/s used in the efficient sup pixel convolutional neural network.
"""
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, upscale_factor=3, channels=1):
        super(Network, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, channels * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pixel_shuffle(x)
        return x