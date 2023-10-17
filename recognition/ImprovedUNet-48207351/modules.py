import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
    
class ContextModule(nn.Module):
    def __init__(self, in_channels, out_channels, pdrop=0.3):
            super(ContextModule, self).__init__()
            self.conv1 = nn.Conv3(in_channels, out_channels, kernel_size=3, padding=1, stride=2)
            self.conv2 = nn.Conv3(in_channels, out_channels, kernel_size=3, padding=1, stride=2)
            self.dropout = nn.Dropout3d(pdrop)
            self.norm = nn.LazyInstanceNorm3d(out_channels)
            self.relu = nn.LeakyReLU(0.01, inplace=True)


    def forward(self, x):
        out = self.norm(self.relu(self.conv1(x)))
        out = self.norm(self.relu(self.conv2(out)))
        out = self.dropout(out)
        return out
    

class LocalizationModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LocalizationModule, self).__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv1 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=1)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.relu = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x, context_features):
        x = self.norm(self.relu(self.upsample(x)))
        # Concatenate with features from the context pathway
        x = torch.cat((x, context_features), dim=1)
        x = self.norm(self.relu(self.conv1(x)))
        x = self.norm(self.relu(self.conv2(x)))
        return x

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

def test():
    x = torch.randn((3, 1, 161, 161))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()