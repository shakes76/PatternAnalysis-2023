import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class FirstStep(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FirstStep, self).__init__()
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        self.ContextModule = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
        )

    def forward(self, x):
        intermediate = self.initial_conv(x)
        out = self.ContextModule(intermediate)
        
        summed_output = intermediate + out
        return summed_output

class MainSteps(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MainSteps, self).__init__()
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        self.ContextModule = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
        )

    def forward(self, x):
        intermediate = self.initial_conv(x)
        out = self.ContextModule(intermediate)
        
        summed_output = intermediate + out
        return summed_output


class UpSteps(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MainSteps, self).__init__()
        
        self.LastCon = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        self.Localization = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        
        out = self.Localization(x)
        out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = self.LastCon(out)
        
        return out