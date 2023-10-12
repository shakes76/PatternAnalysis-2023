import torch
import torch.nn as nn
import torch.nn.functional as F



class uNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(uNet, self).__init__()

        self.conv1 = Conv(in_channels, 16, stride=1, kernel_size=3)
        self.context1 = ResBlock(16)
        self.conv2 = Conv(16,32, stride=2, kernel_size=3)
        self.context2 = ResBlock(32)
        self.conv3 = Conv(32,64, stride=2, kernel_size=3)
        self.context3 = ResBlock(64)
        self.conv4 = Conv(64,128, stride=2, kernel_size=3)
        self.context4 = ResBlock(128)
        self.conv5 = Conv(128,256,stride=2, kernel_size=3)
        self.context5 = ResBlock(256)
        self.up1 = UpSample(256,128)
        self.loc1 = localization(256, 128)
        self.up2 = UpSample(128,64)
        self.loc2 = localization(128,64)
        self.up3 = UpSample(64,32)
        self.loc3 = localization(64,32)
        self.up4 = UpSample(32,16)
        self.conv6 = Conv(32,32, stride=1, kernel_size=3)
        
        #segmentation layers
        self.upsample = nn.Upsample(scale_factor=2)
        self.seg3conv = Conv(64,3, stride=1, kernel_size=1)
        self.seg2conv = Conv(32,3, stride=1, kernel_size=1)
        self.seg1conv = Conv(32,3, stride=1, kernel_size=1)
        self.softmax = nn.Softmax(dim = 3)

    def forward(self, x):
        #Layer 1 downward // input:3 output:16
        lay1d = self.conv1(x)
        lay1d = self.context1(lay1d)

        #Layer 2 downward // input: 16 output: 32
        lay2d = self.conv2(lay1d)
        lay2d = self.context2(lay2d)

        #Layer 3 downward // input:32 output:64
        lay3d = self.conv3(lay2d)
        lay3d = self.context3(lay3d)

        #Layer 4 downward // Input: 64, Output: 128
        lay4d = self.conv4(lay3d)
        lay4d = self.context4(lay4d)

        #Bottom layer 5 // Input: 128, Output: 128
        lay5 = self.conv5(lay4d)
        lay5 = self.context5(lay5)
        lay5 = self.up1(lay5)

        #Layer 4 upwards // Input 256, Output: 64
        lay4u = torch.concat(lay4d, lay5)
        lay4u = self.loc1(lay4u)
        lay4u = self.up2(lay4u)

        #Layer 3 upwards // Input: 128, Output: 32
        lay3u = torch.concat(lay3d, lay4u)
        seg3 = self.loc2(lay3u)
        lay3u = self.up3(seg3)

        #Layer 2 upwards // Input: 64, Output: 16
        lay2u = torch.concat(lay2d, lay3u)
        seg2 = self.loc3(lay2u)
        lay2u = self.up4(seg2)

        #Layer 1 upwards // Input 32, Output: 32
        lay1u = torch.concat(lay1d, lay2u)
        lay1u = self.conv6(lay1u)

        #Segment layers Input: 64 or 32, Output 3
        seg3 = self.seg3conv(seg3)
        seg2 = self.seg2conv(seg2)
        seg1 = self.seg1conv(lay1u)
        
        seg2 = torch.sum(seg2, self.upsample(seg3))
        seg1 = torch.sum(seg1, self.upsample(seg2))

        out = self.softmax(seg1)

        return out




class ResBlock(nn.Module):
    def __init__(self, plane):
        super(ResBlock, self).__init__()
        self.bn1 = nn.InstanceNorm3d(plane)
        self.conv1 = nn.Conv3d(plane, plane, kernel_size=3)
        self.drop = nn.Dropout3d(p=0.3)

        self.bn2 = nn.InstanceNorm3d(plane)
        self.conv2 = nn.Conv3d(plane, plane, kernel_size=3)
          

    def forward(self, x):
        out = self.conv1(F.leaky_relu(self.bn1(x), negative_slope=10**-2))
        out = self.drop(out)
        out = self.conv2(F.leaky_relu(self.bn2(out), negative_slope=10**-2))

        return torch.sum(out, x)


       
class Conv(nn.Module):
    def __init__(self, in_plane, out_plane, stride, kernel_size):
        super().__init__()
        self.conv = nn.Sequential(
                nn.Conv3d(in_plane, out_plane, stride=stride, kernel_size=kernel_size),
                nn.InstanceNorm3d(out_plane),
                F.leaky_relu(negative_slope=10**-2)
        )

    def forward(self, x):
        return self.conv(x)


class UpSample(nn.Module):
    def __init__(self, in_plane, out_plane):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2)
        self.conv = Conv(in_plane, out_plane, stride=1, kernel_size=3)

    def forward(self, x):
        out = self.up(x)
        out = self.conv(out)

        return out
        

class localization(nn.Module):
    def __init__(self, in_plane, out_plane):
        super().__init__()
        self.conv1 = Conv(in_plane, in_plane, stride=1, kernel_size=3)
        self.conv2 = Conv(in_plane, out_plane, stride=1, kernel_size=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        return out
        