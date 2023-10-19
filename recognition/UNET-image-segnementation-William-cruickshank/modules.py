import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


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
