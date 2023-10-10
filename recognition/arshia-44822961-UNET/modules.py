# Modules.py contains modules necessary for building the model. 

import torch
import torch.nn as nn


# Building blocks of the unet. 
class StandardConv(nn.Module):
    # inherits base class from pytorch modules 
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(StandardConv, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.relu(x)
        return x

