"""
DEFINITION OF THE MODEL
"""
# Imports
import torch
import torch.nn as nn

# Model Definition
class ESPCN(nn.Module):

    def __init__(self, in_channels, upscaling_factor=2):
        super(ESPCN, self).__init__()

        self.activation = nn.Tanh()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, in_channels * (upscaling_factor ** 2), kernel_size=3, padding=1)
        self.out = lambda x: nn.functional.pixel_shuffle(x, upscaling_factor)

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