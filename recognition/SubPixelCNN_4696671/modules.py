"""
DEFINITION OF THE MODEL
"""
# Imports
import torch
import torch.nn as nn

# Model Definition
class ESPCN(nn.Module):

    def __init__(self, in_channels, upscaling_factor=4):
        super(ESPCN, self).__init__()

        self.activation = nn.Tanh()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3)
        self.conv4 = nn.Conv2d(32, in_channels * (upscaling_factor ** 2), kernel_size=3)
        self.out = nn.functional.pixel_shuffle

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.activation(x)

        x = self.conv3(x)
        x = self.activation(x)

        x = self.conv4(x)
        x = self.activation(x)

        x = self.out(x)
        return x