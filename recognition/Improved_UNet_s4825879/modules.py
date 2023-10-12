import torch
import torch.nn as nn
import torch.nn.functional as F

class Context(nn.module):
    def __init__(self, size ,features):
        super(Context, self).__init__()
        self.pdrop = 0.3
        self.neagative_slope = 10**(-2)

        self.instNorm = nn.InstanceNorm3d(features)
        self.conv = nn.Conv3d(
            size, size, kernel_size=3
        )
        self.dropOut = nn.Dropout3d(self.pdrop)

    def forward(self, input):
        out = input
        out = F.leaky_relu(self.instNorm(out), self.neagative_slope)
        out = self.conv(out)
        out = self.dropOut(out)
        out = self.conv(out)
        out = F.leaky_relu(self.instNorm(out), self.neagative_slope)
        return torch.sum(out, input)

class ImpUNet(nn.module):
    def __init__(self, in_channel):
        super(ImpUNet, self).__init__()

        # arcitecture components
        self.conv1 = nn.Conv3d(
            in_channel, 16, kernel_size=3
        )
