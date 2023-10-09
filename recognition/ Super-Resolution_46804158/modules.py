import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

upscale_factor = 3

class SuperResolutionModel(nn.Module):
    def __init__(self, upscale_factor=3, channels=3):
            super(SuperResolutionModel, self).__init__()
            self.conv_args = {
                "kernel_size": 3,
                "padding": 1,
            }
            self.inputs = nn.Conv2d(channels, 64, kernel_size=5, padding=2)
            self.conv1 = nn.Conv2d(64, 64, **self.conv_args)
            self.conv2 = nn.Conv2d(64, 32, **self.conv_args)
            self.conv3 = nn.Conv2d(32, channels * (upscale_factor ** 2), **self.conv_args)

    def forward(self, out):
        out = torch.relu(self.inputs(out))
        out = torch.relu(self.conv1(out))
        out = torch.relu(self.conv2(out))
        out = self.conv3(out)
        out = torch.nn.functional.pixel_shuffle(out, upscale_factor)
        return out
