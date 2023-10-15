import torch
import torch.nn as nn
import torch.nn.functional as F

def adjust_residual(x, residual):
    x_channels = x.shape[1]
    residual_channels = residual.shape[1]

    if x_channels != residual_channels:
        adjustment_layer = nn.Conv2d(residual_channels,x_channels,kernel_size=1).to(x.device)
        residual = adjustment_layer(residual)

    return x + residual

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=0.3)
        self.instancenorm = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        residual = x
        x = F.leaky_relu(self.instancenorm(self.conv1(x)), negative_slope=1e-2)
        x = self.dropout(x)
        x = F.leaky_relu(self.instancenorm(self.conv2(x)), negative_slope=1e-2)
        x = adjust_residual(x, residual)
        return x
    
class LocalizationModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LocalizationModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels // 2, kernel_size=1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope= 1e-2)
        x = F.leaky_relu(self.conv2(x), negative_slope = 1e-2)
        return x

class UNETImproved(nn.Module):
    def __init__(self, n_classes):
        super(UNETImproved, self).__init__()

        # The encoding path
        self.encode1 = ResidualBlock(3, 64)
        self.encode2 = ResidualBlock(64, 128)
        self.encode3 = ResidualBlock(128, 256)
        self.encode4 = ResidualBlock(256, 512)
        self.encode5 = ResidualBlock(512, 1024)

        # The decoding path 
        self.decode1 = LocalizationModule(1024 + 512, 256)
        self.decode2 = LocalizationModule(256 + 128, 128)
        self.decode3 = LocalizationModule(128 + 64, 64)
        self.decode4 = LocalizationModule(64 + 32, 32)

        # # Output
        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):

        # Encode
        e1 = self.encode1(x)
        e2 = self.encode2(F.max_pool2d(e1, 2))
        e3 = self.encode3(F.max_pool2d(e2, 2))
        e4 = self.encode4(F.max_pool2d(e3, 2))
        e5 = self.encode5(F.max_pool2d(e4, 2))

        # Decode
        d1 = F.interpolate(e5, scale_factor=2, mode='nearest')
        d1 = torch.cat([d1, e4], dim=1)
        d1 = self.decode1(d1)

        d2 = F.interpolate(d1, scale_factor=2, mode='nearest')
        d2 = torch.cat([d2, e3], dim=1)
        d2 = self.decode2(d2)

        d3 = F.interpolate(d2, scale_factor=2, mode='nearest')
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.decode3(d3)

        d4 = F.interpolate(d3, scale_factor=2, mode='nearest')
        d4 = torch.cat([d4, e1], dim=1)
        d4 = self.decode4(d4)

        # Output
        out = self.final_conv(d4)
        return torch.sigmoid(out)
    

