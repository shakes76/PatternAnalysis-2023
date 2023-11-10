import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextModule(nn.Module):
    def __init__(self, channels):
        super(ContextModule, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=0.3)
        

    def forward(self, x):
        pre_convolution_x = x
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        return x + pre_convolution_x
    
    
class UpsampleModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # Double Voxel Features
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = F.relu(self.conv1(x))
        return x
        

class LocalizationModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LocalizationModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x

class UNETImproved(nn.Module):
    def __init__(self):
        super(UNETImproved, self).__init__()
        self.init_conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.down_conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.down_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.down_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.down_conv4 = nn.Conv2d(128, 256, kernel_size=3,stride=2,padding=1)
        
        self.context_module1 = ContextModule(16)
        self.context_module2 = ContextModule(32)
        self.context_module3 = ContextModule(64)
        self.context_module4 = ContextModule(128)
        self.context_module5 = ContextModule(256)

        self.upsample_module1 = UpsampleModule(256, 128)
        self.upsample_module2 = UpsampleModule(128, 64)
        self.upsample_module3 = UpsampleModule(64, 32)
        self.upsample_module4 = UpsampleModule(32, 16)

        self.localization_module1 = LocalizationModule(256, 128)
        self.localization_module2 = LocalizationModule(128, 64)
        self.localization_module3 = LocalizationModule(64, 32)

        self.segmentation_layer1 = nn.Conv2d(64, 1, kernel_size=3, padding = 1)
        self.segmentation_layer2 = nn.Conv2d(32, 1, kernel_size=3, padding = 1)
        self.segmentation_layer3 = nn.Conv2d(16, 1, kernel_size=3, padding = 1)

        self.second_last_conv = nn.Conv2d(32, 16, kernel_size = 3, padding = 1)
        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # Encoding Pathway

        # Initial Convolution
        x1 = F.relu(self.init_conv(x))

        # Context Module 1
        x2 = self.context_module1(x1)

        # Stride 2 Convolution
        x3 = self.down_conv1(x2)

        # Context Module 2
        x4 = self.context_module2(x3)

        # Stride 2 Convolution
        x5 = self.down_conv2(x4)

        # Context Module 3
        x6 = self.context_module3(x5)

        # Stride 2 Convolution
        x7 = self.down_conv3(x6)

        # Context Module 4
        x8 = self.context_module4(x7)

        # Stride 2 Convolution
        x9 = self.down_conv4(x8)

        # Context Module 5
        x10 = self.context_module5(x9)

        # Decoding Pathway 

        # Upsample Module 1
        x = self.upsample_module1(x10)
        x = torch.cat((x, x8),dim=1)

        # Localization Module 1
        x = self.localization_module1(x)

        # Upsample Module 2
        x = self.upsample_module2(x)
        x = torch.cat((x, x6),dim=1)

        # Localization Module 2
        x = self.localization_module2(x)

        # Segmentation Layer 1
        seg1 = self.segmentation_layer1(x)

        # Upsample Module 3
        x = self.upsample_module3(x)
        x = torch.cat((x,x4),dim=1)

        # Localization Module 3
        x = self.localization_module3(x)

        # Segmentation Layer 2
        seg2 = self.segmentation_layer2(x) 

        seg1_upsample = F.interpolate(seg1, size=seg2.size()[2:], mode='bilinear',align_corners=True)
        seg2 = seg2 + seg1_upsample

        # Upsample Module 4
        x = self.upsample_module4(x)
        x = torch.cat((x, x2), dim=1)

        x = self.second_last_conv(x)

        # Segmentation Layer 3
        seg3 = self.segmentation_layer3(x) 

        seg2_upsampled = F.interpolate(seg2, size=seg3.size()[2:], mode='bilinear', align_corners=True)
        seg3 = seg3 + seg2_upsampled

        x  = x + seg3

        x = self.final_conv(x)

        x = self.sigmoid(x)
        return x

    