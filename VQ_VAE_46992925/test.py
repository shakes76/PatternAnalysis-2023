import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np  
from tqdm import tqdm
from PIL import Image
import torch.utils.data
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt

conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)

weights = conv.weight.data
biases = conv.bias.data

print("Weights: ", weights)
print("Biases: ", biases.shape)


class MaskedConv2d(nn.Conv2d):


    def __init__(self, num_channels, kernel_size):

        super(MaskedConv2d, self).__init__(num_channels, num_channels, kernel_size=kernel_size, padding=(kernel_size//2))

        #self.register_buffer('mask', torch.zeros_like(self.weight))

        k = self.kernel_size[0]

        self.weight.data[:, :, (k//2+1):, :].zero_()
        self.weight.data[:, :, k//2, k//2:].zero_()


    def forward(self, x):
        k = self.kernel_size[0]
        # Type 'A' mask
        self.weight.data[:, :, (k//2+1):, :].zero_()
        self.weight.data[:, :, k//2, k//2:].zero_()

        out = super(MaskedConv2d, self).forward(x)
        return out
    


masked_conv = MaskedConv2d(num_channels=1, kernel_size=5)

print("Masked weights: ", masked_conv.weight.data)