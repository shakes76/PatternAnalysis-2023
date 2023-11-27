import torch
import torch.nn as nn
import torch.nn.functional as TF
import torchvision
# Double Convelution Module
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        # sequential model
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False), # convolution operation
            nn.BatchNorm2d(out_channels), # batch normalization
            nn.ReLU(inplace=True), # ReLu activation function
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False), # convolution operation
            nn.BatchNorm2d(out_channels), # batch normalization
            nn.ReLU(inplace=True), # ReLu activation function
        )

    # performs double convolution 
    def forward(self, x):
        return self.double_conv(x)

# Improved Unet Module
class ImprovedUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(ImprovedUNet, self).__init__()

        # Get list of down sampled layers (Enocder components)
        self.down_samples = nn.ModuleList() # list of down sampling modules
        channel_in = in_channels
        for feature in features:
            self.down_samples.append(DoubleConv(channel_in, feature))
            channel_in = feature

        # bottleneck layer 
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Get list of up sampled layers (Decoder components)
        self.up_samples = nn.ModuleList() # list of up sampling modules
        for feature in reversed(features):
            self.up_samples.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)) # transposed convolution to upsample
            self.up_samples.append(DoubleConv(feature*2, feature))

        # Segmentation layer (get single out channel)
        self.last_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        # max pool module
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 

    # Perform processing of sample 'x'
    def forward(self, x):
        skip_connections = [] # list of skip connections

        # Down sample sample
        for ds in self.down_samples:
            x = ds(x) 
            skip_connections.append(x) # store skip connect sample
            x = self.pool(x) # max pool downsample

        # Get bottleneck sample
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Up sample sample
        for i in range(0, len(self.up_samples), 2):
            x = self.up_samples[i](x)
            skip_connection = skip_connections[i//2]
            # check size of sample
            if x.shape != skip_connection.shape:
                x = TF.interpolate(x, size=skip_connection.shape[2:]) #
            skip_x = torch.cat((skip_connection, x), dim=1) # concatinate skip and sample along 2-D
            x = self.up_samples[i+1](skip_x) # upsample sample

        # return single channel output binary mask (segmentation layer)
        return self.last_conv(x)
