import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ContextModule, self).__init__()
        # A 3x3 convolution layer followed by instance normalization and leaky ReLU activation
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
        )
        # Another 3x3 convolution layer followed by instance normalization and leaky ReLU activation
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
        )
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout2d(p=0.3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        return x

class LocalisationModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LocalisationModule, self).__init__()
        # Using a simple upscale by repeating the feature pixels twice
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # 3x3 convolution to process concatenated features
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 1x1 convolution to reduce the number of feature maps
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # x = self.upsample(x)
        x = self.conv1(x)
        x = F.leaky_relu(x, negative_slope=1e-2)
        x = self.conv2(x)
        x = F.leaky_relu(x, negative_slope=1e-2)
        return x

class UpsamplingModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsamplingModule, self).__init__()
        # Using a simple upscale by repeating the feature pixels twice
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # 3x3 convolution that halves the number of feature maps
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = F.leaky_relu(x, negative_slope=1e-2)
        return x

class SegmentationLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegmentationLayer, self).__init__()
        # A convolutional layer that produces segmentation map
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        # Applying convolution
        x = self.conv(x)
        # Applying sigmoid activation to squash outputs between 0 and 1
        x = torch.sigmoid(x)
        return x
    
class UpscalingLayer(nn.Module):
    def __init__(self, scale_factor=2, mode='nearest'):
        super(UpscalingLayer, self).__init__()
        # An upsampling layer that increases the spatial dimensions of the feature map
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
        
    def forward(self, x):
        # Applying upscaling
        x = self.upsample(x)
        return x
    
class UNet2D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UNet2D, self).__init__()

        # Encoder
        self.enc1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.context1 = ContextModule(16, 16)
        self.enc2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.context2 = ContextModule(32, 32)
        self.enc3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.context3 = ContextModule(64, 64)
        self.enc4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.context4 = ContextModule(128, 128)

        # Bottleneck
        self.bottleneck = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bottleneck_context = ContextModule(256, 256)
        self.up_bottleneck = UpsamplingModule(256, 128)
        
        # Decoder
        self.local1 = LocalisationModule(256, 128)
        self.up1 = UpsamplingModule(128, 64)

        self.local2 = LocalisationModule(128, 64)
        self.up2 = UpsamplingModule(64, 32)

        self.seg1 = SegmentationLayer(64, num_classes)
        self.upsample_seg1 = UpscalingLayer()

        self.local3 = LocalisationModule(64, 32)
        self.up3 = UpsamplingModule(32, 16)

        self.seg2 = SegmentationLayer(32, num_classes)
        self.upsample_seg2 = UpscalingLayer()

        self.final_conv = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.seg3 = SegmentationLayer(32, num_classes)
        self.upsample_seg3 = UpscalingLayer()

    def forward(self, x):
        y1 = self.enc1(x)
        x1 = self.context1(y1)
        x1 = x1 + y1

        y2 = self.enc2(x1)
        x2 = self.context2(y2)
        x2 = x2 + y2

        y3 = self.enc3(x2)
        x3 = self.context3(y3)
        x3 = x3 + y3
        
        y4 = self.enc4(x3)
        x4 = self.context4(y4)
        x4 = x4 + y4

        # Bottleneck
        bottleneck_conv = self.bottleneck(x4)

        bottleneck = self.bottleneck_context(bottleneck_conv)
        bottleneck = bottleneck + bottleneck_conv

        up_bottleneck = self.up_bottleneck(bottleneck)
        
        # Decoder
        x = self.local1(torch.cat((x4, up_bottleneck), 1))
        x = self.up1(x)

        x = self.local2(torch.cat((x3, x), 1))
        seg1 = self.seg1(x)
        x = self.up2(x)

        seg1_upsampled = self.upsample_seg1(seg1)

        x = self.local3(torch.cat((x2, x), 1))
        seg2 = self.seg2(x)
        x = self.up3(x)

        seg12 = seg1_upsampled + seg2
        seg12_up = self.upsample_seg2(seg12)
        
        x = self.final_conv(torch.cat((x1, x), 1))

        seg3 = self.seg3(x)
        seg123 = seg3 + seg12_up

        out = nn.functional.softmax(seg123, dim=1)
        # print(out.shape)
        
        return out
