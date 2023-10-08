import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision
import numpy as np
import PIL.Image as Image
import PIL
import os

class DoubleConv(nn.Module):
    """
    A module consisting of two convolutional layers with batch normalization and ReLU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    
    def __init__(self, in_channels=3, out_channels=1, features=[64,128,256,512]):
        super().__init__()
        self.ups = nn.ModuleList()   # list of all upsampling layers
        self.downs = nn.ModuleList() # list of all downsampling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Downsampling part of the UNet
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Upsampling part of the UNet
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, # *2 because of skip connections
                                               feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))
        
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        
        # Downsample the input
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck layer
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] # reverse the order of skip connections
        
        # Upsample the output
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2] # //2 because we are taking 2 steps at a time
            
            # if the dimensions of the skip connection and the upsampled output don't match,
            # then crop the skip connection to match the dimensions of the output
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
            
        return self.final_conv(x)

def test():
    x = torch.randn((3, 1, 256, 256))
    model = UNet(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape

if __name__ == '__main__':
    test()