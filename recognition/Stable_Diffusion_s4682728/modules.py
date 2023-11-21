from imports import *

class SinusoidalPositionEmbeddings(nn.Module):
    '''
    Sourced from: https://huggingface.co/blog/annotated-diffusion
    Author: Niels Rogge, Kashif Rasul
    '''
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# Define a class for a residual network block
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb=32):
        super(ResNetBlock, self).__init__()
        self.time_mlp =  nn.Linear(time_emb, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
        self.shortcut = nn.Sequential() # Shortcut for identity mapping
        if in_channels != out_channels:
            # Adjust dimensions if necessary
            self.shortcut.add_module('conv_shortcut', nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0))
            self.shortcut.add_module('bn_shortcut', nn.BatchNorm2d(out_channels))

    def forward(self, x, t):
        residual = self.shortcut(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        time_emb = self.time_mlp(t)
        time_emb = self.relu(time_emb)
        time_emb = time_emb[(..., ) + (None,) * 2]
        x = x + time_emb
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        x += residual
        x = self.relu(x)
        
        return x

# Define a class for an encoder block
class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, time_emb=32):
        super().__init__()
        self.blocks = nn.ModuleList([ResNetBlock(in_c if i == 0 else out_c, out_c) for i in range(num_blocks)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t):
        for block in self.blocks:
            x = block(x, t)
        skip = x.clone() # Keep a copy for skip connection
        x = self.pool(x)
        return x, skip

# Define a class for a decoder block
class DecoderBlock(nn.Module):
    def __init__(self, num_in, num_out, num_blocks=2):
        super().__init__()
        self.up = nn.ConvTranspose2d(num_in, num_out, kernel_size=2, stride=2, padding=0)
        self.blocks = nn.ModuleList([ResNetBlock(num_out * 2 if i == 0 else num_out, num_out) for i in range(num_blocks)])

    def forward(self, x, t, skip):
        x = self.up(x)
        x = torch.cat([x, skip], axis=1) # Concatenate with the skip connection
        for block in self.blocks:
            x = block(x, t)
        return x

# Define the main class for the diffusion network
class DiffusionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder blocks
        self.down1 = EncoderBlock(1, 64)
        self.down2 = EncoderBlock(64, 128)
        self.down3 = EncoderBlock(128, 256)
        self.down4 = EncoderBlock(256, 512)
        
        # Bottleneck block
        self.bottle_neck = ResNetBlock(512, 1024)
        
        # Decoder blocks
        self.up1 = DecoderBlock(1024, 512)
        self.up2 = DecoderBlock(512, 256)
        self.up3 = DecoderBlock(256, 128)
        self.up4 = DecoderBlock(128, 64)
        
        # Batch normalization and output layer
        self.norm_out = nn.BatchNorm2d(64) 
        self.out = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        
        # Time embedding layers
        time_dim = 32
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU(),
        )

    def forward(self, x, t):
        t = self.time_mlp(t)
        
        residuals = []  # Keep the skip connections
        
        # Downsample
        x, skip1 = self.down1(x, t)
        residuals.append(skip1)
        
        x, skip2 = self.down2(x, t)
        residuals.append(skip2)
        
        x, skip3 = self.down3(x, t)
        residuals.append(skip3)
        
        x, skip4 = self.down4(x, t)
        residuals.append(skip4)
        
        # Bottleneck
        x = self.bottle_neck(x, t)
        
        # Upsample
        x = self.up1(x, t, residuals.pop())
        x = self.up2(x, t, residuals.pop())
        x = self.up3(x, t, residuals.pop())
        x = self.up4(x, t, residuals.pop())
        
        # Normalisation
        x = self.norm_out(x)
        
        return self.out(x)
