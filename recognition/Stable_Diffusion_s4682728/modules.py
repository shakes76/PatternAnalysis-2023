import torch
import torch.nn as nn

class DiffusionProcess(nn.Module):
    def __init__(self, betas, num_steps):
        super(DiffusionProcess, self).__init__()
        self.betas = betas  # List of beta values for each time step
        self.num_steps = num_steps

    def forward(self, x):
        for step in range(self.num_steps):
            beta = self.betas[step].clone().detach()
            noise = torch.randn_like(x) * torch.sqrt(beta)
            x = x + noise
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# class Block(nn.Module):
#     def __init__(self, in_c, out_c, dropout=0.0):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
#         self.bnorm1 = nn.BatchNorm2d(out_c)
#         self.bnorm2 = nn.BatchNorm2d(out_c)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
#         self.attention = ChannelAttention(out_c)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bnorm1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.conv2(x)
#         x = self.bnorm2(x)
#         x = self.attention(x)
#         x = self.relu(x)
#         return x

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.attention = ChannelAttention(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('conv_shortcut', nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0))
            self.shortcut.add_module('bn_shortcut', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.attention(out)
        
        out += residual
        out = self.relu(out)
        
        return out

# Modified Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([ResNetBlock(in_c if i == 0 else out_c, out_c) for i in range(num_blocks)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        skip = x.clone()
        x = self.pool(x)
        return x, skip

# Modified Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self, num_in, num_out, num_blocks=2):
        super().__init__()
        self.up = nn.ConvTranspose2d(num_in, num_out, kernel_size=2, stride=2, padding=0)
        self.blocks = nn.ModuleList([ResNetBlock(num_out * 2 if i == 0 else num_out, num_out) for i in range(num_blocks)])

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], axis=1)
        for block in self.blocks:
            x = block(x)
        return x

# Diffusion Network with U-Net architecture
class DiffusionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        down_channels = (1, 64, 128, 256, 512)
        up_channels = (1024, 512, 256, 128, 64)

        self.downs = nn.ModuleList([EncoderBlock(down_channels[i], down_channels[i+1]) for i in range(len(down_channels)-1)])
        self.ups = nn.ModuleList([DecoderBlock(up_channels[i], up_channels[i+1]) for i in range(len(up_channels)-1)])
        
        self.bottle_neck = ResNetBlock(down_channels[-1], up_channels[0])

        self.norm_out = nn.BatchNorm2d(up_channels[-1])  # Feature normalization
        self.out = nn.Conv2d(up_channels[-1], 1, kernel_size=1, padding=0)

    def forward(self, x):
        residuals = []
        for down in self.downs:
            x, skip = down(x)
            residuals.append(skip)

        x = self.bottle_neck(x)

        for up in self.ups:
            x = up(x, residuals.pop())

        x = self.norm_out(x)  # Feature normalization before the output layer
        return self.out(x)
