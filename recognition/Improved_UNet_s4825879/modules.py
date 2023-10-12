import torch
import torch.nn as nn
import torch.nn.functional as F

class Context(nn.module):
    def __init__(self, size):
        super(Context, self).__init__()
        self.pdrop = 0.3
        self.neagative_slope = 10**(-2)

        self.instNorm = nn.InstanceNorm3d(size)
        self.conv = nn.Conv3d(
            size, size, kernel_size=3
        )
        self.dropOut = nn.Dropout3d(self.pdrop)

    def forward(self, input):
        out = input
        out = F.leaky_relu(self.instNorm(out), self.neagative_slope)
        out = self.dropOut(out)
        out = F.leaky_relu(self.instNorm(out), self.neagative_slope)
        return torch.sum(out, input)

class Upsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsampling, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2)
        self.instNorm = nn.InstanceNorm3d(out_channels)
        self.conv = nn.Conv3d(
           in_channels, out_channels, kernel_size=3 
        ) 

    def forward(self, out):
        out = self.upsample(out)
        out = F.leaky_relu(self.instNorm(self.conv(out)), 10**(-2))
        return out
        
class localization(nn.Module):
    def __init__(self, in_channels):
        super(localization, self).__init__()

        self.conv1 = nn.Conv3d(
            in_channels, in_channels, kernel_size=3
        )
        self.instNorm1 = nn.InstanceNorm3d(in_channels)
        self.conv2 = nn.Conv3d(
            in_channels, in_channels/2, kernel_size=1
        )
        self.instNorm2 = nn.InstanceNorm3d(in_channels/2)
        
    def forward(self, out):
        out = F.leaky_relu(self.instNorm1(self.conv1(out)), 10**(-2))
        out = F.leaky_relu(self.instNorm2(self.conv2(out)), 10**(-2)) 
        return out
        
class ImpUNet(nn.module):
    def __init__(self, in_channel):
        super(ImpUNet, self).__init__()

        # arcitecture components
        self.conv1 = nn.Conv3d(
            in_channel, 16, kernel_size=3
        )
        self.context1 = Context(16)

        self.conv2 = nn.Conv3d(
            16, 32, kernel_size=3, stride=2
        )
        self.context2 = Context(32)

        self.conv3 = nn.Conv3d(
            32, 64, kernel_size=3, stride=2
        )
        self.context3 = Context(64)
        
        self.conv4 = nn.Conv3d(
            64, 128, kernel_size=3, stride=2
        )
        self.context4 = Context(128)
        
        self.conv5 = nn.Conv3d(
            128, 256, kernel_size=3, stride=2
        )
        self.context5 = Context(256)
        
