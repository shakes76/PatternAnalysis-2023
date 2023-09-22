import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(2, stride=2)
        )
        
    def forward(self, x):
        return self.conv(x)
    
class SiameseNetwork(nn.Module):
    def __init__(self, layers : tuple = [1, 32, 64, 128, 256]):
        super(SiameseNetwork, self).__init__()
        self.layers = layers
        self.blocks = nn.ModuleList([ConvBlock(layers[i], layers[i+1]) for i in range(len(layers) - 1)])
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(layers[-1] * 16 * 15, 256)
        )
        
    def forward_once(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.linear(x)
        x = F.sigmoid(x)
        return x
    
    def forward(self, x1, x2):
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)
        return output1, output2
    