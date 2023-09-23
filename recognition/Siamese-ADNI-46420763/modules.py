import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        
    def forward(self, x):
        return self.conv(x)
    
class SiameseNetwork(nn.Module):
    def __init__(self, layers = [1, 32, 64, 128], kernel_sizes = [10, 7, 4, 4]):
        super(SiameseNetwork, self).__init__()
        self.layers = layers
        self.blocks = nn.ModuleList([ConvBlock(layers[i], layers[i+1], kernel_size=kernel_sizes[i]) for i in range(len(layers) - 1)])
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*25*27, 512)
        )
        self.dense = nn.Linear(512, 1)
        
    def forward_once(self, x):
        for block in self.blocks:
            x = block(x)
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
    