import torch
import torch.nn as nn

class ContextBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pdrop=0.3):
        super(ContextBlock, self).__init__()

        # Pre-Activation Residual Block with two 3x3 convolutional layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        
        # Dropout layer
        self.dropout = nn.Dropout2d(p=pdrop)

    def forward(self, x):
        # Pre-Activation Residual Block
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        return x
    
class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(UpsamplingBlock, self).__init__()
        
        # Upsampling layer
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Convolutional layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
                
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
                
        return x
    
class LocalisationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(LocalisationBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        
        return x

class UNetImproved(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(UNetImproved, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
        # Encoder
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=kernel_size, padding=1)
        self.context1 = ContextBlock(16, 16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=kernel_size, padding=1, stride=2)
        self.context2 = ContextBlock(32, 32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=kernel_size, padding=1, stride=2)
        self.context3 = ContextBlock(64, 64)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=kernel_size, padding=1, stride=2)
        self.context4 = ContextBlock(128, 128)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=kernel_size, padding=1, stride=2)
        self.context5 = ContextBlock(256, 256)
        
        self.up1 = UpsamplingBlock(256, 128)
        self.local1 = LocalisationBlock(256, 128)
        
        self.up2 = UpsamplingBlock(128, 64)
        self.local2 = LocalisationBlock(128, 64)
        self.seg1 = nn.Conv2d(64, out_channels, stride=1, kernel_size=kernel_size, padding=1)
        
        self.up3 = UpsamplingBlock(64, 32)
        self.local3 = LocalisationBlock(64, 32)
        self.seg2 = nn.Conv2d(32, out_channels, stride=1, kernel_size=kernel_size, padding=1)
        
        self.up4 = UpsamplingBlock(32, 16)
        self.local4 = LocalisationBlock(32, 16)
        
        self.conv6 = nn.Conv2d(16, 32, kernel_size=kernel_size, padding=1)
        
        self.seg3 = nn.Conv2d(32, out_channels, stride=1, kernel_size=kernel_size, padding=1)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.relu(x)
        y = self.context1(x)
        skip_1 = x + y
        
        x = self.conv2(skip_1)
        x = self.relu(x)
        y = self.context2(x)
        skip_2 = x + y
        
        x = self.conv3(skip_2)
        x = self.relu(x)
        y = self.context3(x)
        skip_3 = x + y
        
        x = self.conv4(skip_3)
        x = self.relu(x)
        y = self.context4(x)
        skip_4 = x + y
        
        x = self.conv5(skip_4)
        x = self.relu(x)
        y = self.context5(x)
        x = x + y
        
        # Decoder
        x = self.up1(x)
        x = torch.cat([x, skip_4], dim=1)
        x = self.local1(x)
        
        x = self.up2(x)
        x = torch.cat([x, skip_3], dim=1)
        y = self.local2(x)
        
        skip_seg_1 = self.seg1(y)
        skip_seg_1 = nn.Upsample(scale_factor=2, mode='nearest')(skip_seg_1)
        
        x = self.up3(y)
        x = torch.cat([x, skip_2], dim=1)
        y = self.local3(x)
        
        skip_seg_2 = self.seg2(y)
        skip_seg_2 = skip_seg_2 + skip_seg_1
        skip_seg_2 = nn.Upsample(scale_factor=2, mode='nearest')(skip_seg_2)
        
        x = self.up4(y)
        x = torch.cat([x, skip_1], dim=1)
        x = self.local4(x)
        
        x = self.conv6(x)
        x = self.relu(x)
        x = self.seg3(x)
        
        x = x + skip_seg_2
        
        x = self.sigmoid(x)
        
        return x
        
        
        
        