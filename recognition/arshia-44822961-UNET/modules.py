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
    

# Pre-activation residual block 
# two 3x3 layers and a drop out layer 
# described as pre-activation res block with 2 convs with drop out layer in between 
# entire feature mapping process using leaky relu as described by the paper. 

# TODO - batch normalisation is incorrect 
# The paper uses instance norm
class ContextModule(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.3):
        super(ContextModule, self).__init__()

        # Batch normalisation before ReLU
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.LeakyReLU(inplace=True)

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        
        # Dropout layer in between 
        self.dropout = nn.Dropout2d(p=dropout_p)

        # Batch normalisation before ReLU
        self.bn2 = nn.BatchNorm2d(out_channels)

        # RELU 
        self.relu2 = nn.LeakyReLU(out_channels)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        
    def forward(self, x):
        # Forward pass through the context module
        # Batch normalisation and ReLU before the first convolution
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        
        return out

# added second conv block with stride 2 
# MESSY??? combine with the other conv block when cleaning up. 
class Conv2dStride2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(Conv2dStride2, self).__init__()

        # Define a 3x3 convolutional layer with stride 2 and optional padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding)

        # Leaky RELU 
        self.relu = nn.LeakyReLU(out_channels)

    def forward(self, x):
        # Forward pass through the convolutional layer
        out = self.conv(x)
        out = self.relu(out)
        return out
    
