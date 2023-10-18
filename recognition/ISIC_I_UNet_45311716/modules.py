import torch
import torch.nn as nn
import torch.nn.functional as TF

# Two Convelutions
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        # sequential model
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False), # convolution operation
            nn.BatchNorm2d(out_channels), # batch normalization
            nn.ReLU(inplace=True), # ReLu activation function
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False), # convolution operation
            nn.BatchNorm2d(out_channels), # batch normalization
            nn.ReLU(inplace=True), # ReLu activation function
        )

    # performs double convolution 
    def forward(self, x):
        return self.double_conv(x)