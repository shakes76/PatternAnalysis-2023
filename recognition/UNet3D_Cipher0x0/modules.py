import torch
import torch.nn as nn
import torch.nn.functional as F


# Construct the model of Unet 3D
# The tensor shape should be (batch, channel, height, width, depth)
# The number of input channels depends on the dataset, e.g Prostate3D has 1
# The number of output channels depends on the dataset, e.g Prostate3D has 6
class UNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv3d(4, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2), nn.ReLU(),
            nn.BatchNorm3d(16))

        self.layer2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2), nn.ReLU(),
            nn.BatchNorm3d(64))

        self.layer3 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2), nn.ReLU(),
            nn.BatchNorm3d(64))

        self.layer4 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2), nn.ReLU(),
            nn.BatchNorm3d(128))

        # Decoder
        self.decode3 = Decoder(128, 64 + 64, 64)
        self.decode2 = Decoder(64, 64 + 64, 64)
        self.decode1 = Decoder(64, 16 + 16, 16)
        self.decode0 = nn.Sequential(
            nn.ConvTranspose3d(16, 16, kernel_size=2, stride=2),
            nn.Conv3d(16, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=3, padding=1), nn.ReLU())
        self.output_layer = nn.Conv3d(16, 6, 1)

    def forward(self, input):
        e1 = self.layer1(input)  # 16,64,128,128
        e2 = self.layer2(e1)  # 64,32,64,64
        e3 = self.layer3(e2)  # 64,16,32,32
        e4 = self.layer4(e3)  # 128,8,16,16

        d3 = self.decode3(e4, e3)  # 64,16,32,32
        d2 = self.decode2(d3, e2)  # 64,32,64,64
        d1 = self.decode1(d2, e1)  # 16,64,128,128
        d0 = self.decode0(d1)  # 16,128,256,256

        out = self.output_layer(d0)  # 6,128,256,256
        return out


class Decoder(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_relu = nn.Sequential(
            nn.Conv3d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm3d(out_channels))

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat((x1, x2), dim=1)  # add shortcut layer and input layer together
        x1 = self.conv_relu(x1)
        return x1


# Use GPU as default
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.empty_cache()
unet = UNet3D().to(device)
