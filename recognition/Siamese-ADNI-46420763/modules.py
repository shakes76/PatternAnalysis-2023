import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    Basic Convlutional Block
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        return self.conv(x)
    
class SiameseNetwork(nn.Module):
    """
    Siamese Network Model
    """
    def __init__(self, layers = [1, 64, 128, 128, 256], kernel_sizes = [10, 7, 4, 4]):
        super(SiameseNetwork, self).__init__()
        self.layers = layers
        self.blocks = nn.ModuleList([ConvBlock(layers[i], layers[i+1], kernel_size=kernel_sizes[i]) for i in range(len(layers) - 1)])
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 7, 256)
        )
        self.dense = nn.Linear(256, 1)
        
    def forward_once(self, x):
        """
        Creates embedding vector of input image
        """
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i != len(self.blocks) - 1:
                x = self.maxpool(x)
        x = self.linear(x)
        x = F.sigmoid(x)
        return x
    
    def forward(self, x1, x2):
        embedding1 = self.forward_once(x1)
        embedding2 = self.forward_once(x2)
        distance = torch.abs(embedding1 - embedding2)
        
        output = self.dense(distance)
        output = F.sigmoid(output)
        return output