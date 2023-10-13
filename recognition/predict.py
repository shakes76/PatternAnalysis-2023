import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, middle_channels=None):
        super().__init__()
        if not middle_channels:
            middle_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class ImprovedUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(ImprovedUNet, self).__init__()

        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        # Channel reducer (Bridge)
        self.bridge = nn.Conv2d(512, 256, kernel_size=1)

        # Decoder
        self.dec1 = DoubleConv(512, 256, 384) # Adjusted the channels
        self.dec2 = DoubleConv(384, 128, 192)
        self.dec3 = DoubleConv(128, 64)
        self.out_conv = nn.Conv2d(64, out_channels, 1)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Reduce channels of e4 to match e3
        e4_reduced = self.bridge(self.up(e4))

        # Decoder (with channel-reduced e4)
        d1 = self.dec1(torch.cat([e4_reduced, e3], 1))
        d2 = self.dec2(torch.cat([self.up(d1), e2], 1))
        d3 = self.dec3(self.up(d2))
        out = self.out_conv(d3)

        #print("Shape of e1:", e1.shape)
        #print("Shape of upsampled e1:", self.up(e1).shape)
        #print("Shape of e2:", e2.shape)
        #print("Shape of upsampled e2:", self.up(e2).shape) # Fixed print label
        #print("Shape of e3:", e3.shape)
        #print("Shape of upsampled e3:", self.up(e3).shape)
        #print("Shape of e4:", e4.shape)
        #print("Shape of upsampled e4:", self.up(e4).shape)

        return out
