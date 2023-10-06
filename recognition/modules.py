import torch
import torch.nn as nn
import torch.nn.functional as F

# Implementing the context module which comprises a pre-activation residual block
class ContextModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ContextModule, self).__init__()
        # A 3x3x3 convolution layer followed by instance normalisation and leaky ReLU activation
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride = 2),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
        )
        # Another 3x3x3 convolution layer followed by instance normalisation and leaky ReLU activation
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride = 2),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
        )
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout3d(p=0.3)

    def forward(self, x):
        # Saving the input x for the skip connection
        identity = x
        # Passing x through the two convolutions
        x = self.conv1(x)
        x = self.conv2(x)
        # Applying dropout
        x = self.dropout(x)
        # Adding the identity (input) to the output (skip connection)
        # x += identity
        return x

# Implementing the localisation module
class LocalisationModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LocalisationModule, self).__init__()
        # 3x3x3 convolution to process concatenated features
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2)
        # 1x1x1 convolution to reduce the number of feature maps
        self.conv2 = nn.Conv3d(out_channels, out_channels//2, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x, negative_slope=1e-2)
        x = self.conv2(x)
        x = F.leaky_relu(x, negative_slope=1e-2)
        return x

# Implementing the upsampling module
class UpsamplingModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsamplingModule, self).__init__()
        # Using a simple upscale by repeating the feature voxels twice
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # 3x3x3 convolution that halves the number of feature maps
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = F.leaky_relu(x, negative_slope=1e-2)
        return x

class UNet3D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UNet3D, self).__init__()
        
        # Encoder
        # Encoder Layer 1
        self.enc1 = nn.Conv3d(in_channels, 16, kernel_size=3, padding=1)
        self.context1 = ContextModule(16, 16)
        
        # Encoder Layer 2
        self.enc2 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)
        self.context2 = ContextModule(32, 32)
        
        # Encoder Layer 3
        self.enc3 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.context3 = ContextModule(64, 64)
        
        # Encoder Layer 4
        self.enc4 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.context4 = ContextModule(128, 128)
        
        # Bottleneck
        self.bottleneck = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bottleneck_context = ContextModule(256, 256)
        self.up_bottleneck = UpsamplingModule(256, 128)
        
        # Decoder
        # Decoding Layer 1
        self.local1 = LocalisationModule(256, 128)
        self.up1 = UpsamplingModule(64, 64)
        
        # Decoding Layer 2
        self.local2 = LocalisationModule(128, 64)
        self.up2 = UpsamplingModule(32, 32)
        
        # Decoding Layer 3
        self.local3 = LocalisationModule(64, 32)
        self.up3 = UpsamplingModule(16, 16)
        
        # Decoding Layer 4
        self.final_conv = nn.Conv3d(32, num_classes, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Encoder
        x1 = self.context1(self.enc1(x))
        print(f"x1 size: {x1.size()}")
        x2 = self.context2(self.enc2(x1))
        print(f"x2 size: {x2.size()}")
        x3 = self.context3(self.enc3(x2))
        print(f"x3 size: {x3.size()}")
        x4 = self.context4(self.enc4(x3))
        print(f"x4 size: {x4.size()}")
        
        # Bottleneck
        bottleneck = self.bottleneck_context(self.bottleneck(x4))
        up_bottleneck = self.up_bottleneck(bottleneck)

        # Decoder
        # Concatenate upsampled bottleneck with encoder output and apply Localisation
        x = self.local1(torch.cat((x4, up_bottleneck), 1))
        x = self.up1(x)
        
        x = self.local2(torch.cat((x3, x), 1))
        x = self.up2(x)
        
        x = self.local3(torch.cat((x2, x), 1))
        x = self.up3(x)
        
        x = self.final_conv(x)
        out = nn.functional.softmax(x, dim=1)  # Apply Softmax along the channel dimension
        
        return out
