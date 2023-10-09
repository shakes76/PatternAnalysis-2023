import torch
import torch.nn as nn

class DiffusionProcess(nn.Module):
    def __init__(self, betas, num_steps):
        super(DiffusionProcess, self).__init__()
        self.betas = betas  # List of beta values for each time step
        self.num_steps = num_steps

    def forward(self, x):
        for step in range(self.num_steps):
            beta = torch.tensor(self.betas[step], dtype=torch.float32)
            noise = torch.randn_like(x) * torch.sqrt(beta)
            x = x + noise
        return x

# Basic building block
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer1 = Block(1, 64, stride=2)
        self.layer2 = Block(64, 128, stride=2)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        return x1, x2

# Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layer1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.layer2 = nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x1, x2):
        x = self.layer1(x2)
        x += x1  # Skip connection
        x = self.layer2(x)
        return x

# UNet using Encoder and Decoder
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x1, x2 = self.encoder(x)
        x = self.decoder(x1, x2)
        return x

# Diffusion Network with U-Net architecture
class DiffusionNetwork(nn.Module):
    def __init__(self, beta, num_steps):
        super(DiffusionNetwork, self).__init__()
        self.beta = beta
        self.num_steps = num_steps
        self.unet = UNet()

    def forward(self, x):
        for step in range(self.num_steps):
            noise = torch.randn_like(x) * self.beta
            x = x + noise
            x = self.unet(x)
        return x




