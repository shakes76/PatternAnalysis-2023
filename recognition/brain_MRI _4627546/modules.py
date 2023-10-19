"""
modules.py

Student Name: Zijun Zhu
Student ID: s4627546
Bref intro:
Containing the source code of the components of the model.
Each component is implemented as a class or a function

reference: https://keras.io/examples/vision/super_resolution_sub_pixel/:    Build a model
"""

import torch.nn as nn
import torch.nn.functional as F


class ESPCN(nn.Module):
    def __init__(self, upscale_factor=4, channels=1):
        """
        upscale_factor: 4x
        channels: input has only one channels
        """
        super(ESPCN, self).__init__()
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=5, padding=2, padding_mode='reflect')
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, padding_mode='reflect')
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1, padding_mode='reflect')
        self.conv4 = nn.Conv2d(32, channels * (upscale_factor ** 2), kernel_size=3, padding=1, padding_mode='reflect')
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x


# Instantiate the model
if __name__ == '__main__':
    model = ESPCN(upscale_factor=4, channels=1)
    print(model)
