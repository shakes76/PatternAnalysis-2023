import torch  # Import PyTorch for deep learning functionalities.
import torch.nn as nn  # Neural network module from PyTorch for creating layers.

# Determine if CUDA (GPU support) is available, use it, otherwise default to CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Modules: {torch.cuda.is_available()}")  # Display whether CUDA is available.


# Defining a double convolution block module.
class DoubleConv(nn.Module):
    # This class represents a sequence of operations that consist of two convolution layers.
    # each followed by a batch normalization and ReLU activation.
    def __init__(self, in_channels, out_channels, middle_channels=None):
        super().__init__()
        if not middle_channels:
            middle_channels = out_channels  # If no middle channels, make it the same as out_channels.

        # Sequential container for all the layers.
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),  # First convolution layer.
            nn.BatchNorm2d(middle_channels),  # Batch Normalization for the first conv layer.
            nn.ReLU(inplace=True),  # ReLU activation in-place.
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),  # Second convolution layer.
            nn.BatchNorm2d(out_channels),  # Batch Normalization for the second conv layer.
            nn.ReLU(inplace=True)  # ReLU activation in-place.
        )

    def forward(self, x):
        # Defines the computation performed at every call, passing the input through all the layers.
        return self.double_conv(x)

class ImprovedUNet(nn.Module):
    # This class represents the U-Net like architecture for semantic segmentation.
    def __init__(self, in_channels=3, out_channels=1):
        super(ImprovedUNet, self).__init__()

        # Encoder path (downsampling).
        self.enc1 = DoubleConv(in_channels, 64)  # First double conv block.
        self.enc2 = DoubleConv(64, 128)  # Second double conv block.
        self.enc3 = DoubleConv(128, 256)  # Third double conv block.
        self.enc4 = DoubleConv(256, 512)  # Fourth double conv block.

        self.pool = nn.MaxPool2d(2)  # Pooling layer that halves the dimensions.

        # Channel reducer (Bridge between encoder and decoder).
        self.bridge = nn.Conv2d(512, 256, kernel_size=1)  # 1x1 Convolution to reduce the number of channels.

        # Decoder path (upsampling)
        self.dec1 = DoubleConv(512, 256, 384)  # First block in decoder with concatenated features.
        self.dec2 = DoubleConv(384, 128, 192)  # Second block in decoder with concatenated features.
        self.dec3 = DoubleConv(128, 64)  # Third block in decoder.
        self.out_conv = nn.Conv2d(64, out_channels, 1)  # 1x1 Convolution for generating final output.

        # Upsampling layer.
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # Increase resolution.

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)  # Pass input through first encoder block
        e2 = self.enc2(self.pool(e1))  # Downsample and pass through second encoder block.
        e3 = self.enc3(self.pool(e2))  # Downsample and pass through third encoder block.
        e4 = self.enc4(self.pool(e3))  # Downsample and pass through fourth encoder block.

        # Reduce channels of e4 to match e3 for the skip connection.
        e4_reduced = self.bridge(self.up(e4))  # Upsample and reduce channels.

        # Decoder with skip connections
        d1 = self.dec1(torch.cat([e4_reduced, e3], 1))  # Concatenate reduced e4 and e3, pass through decoder block.
        d2 = self.dec2(torch.cat([self.up(d1), e2], 1))  # Upsample, concatenate with e2, pass through decoder block.
        d3 = self.dec3(self.up(d2))  # Upsample and pass through decoder block.
        out = self.out_conv(d3)  # Get the final output.

        return out 
