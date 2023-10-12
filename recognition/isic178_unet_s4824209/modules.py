import torch
import torch.nn as nn
import torch.nn.functional as F



class uNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(uNet, self).__init__()

        self.conv1 = Conv(in_channels, 16, stride=1)
        self.context1 = ResBlock(16)
        self.conv2 = Conv(16,32, stride=2)
        self.context2 = ResBlock(32)
        self.conv3 = Conv(32,64, stride=2)
        self.context3 = ResBlock(64)
        self.conv4 = Conv(64,128, stride=2)
        self.context4 = ResBlock(128)
        self.conv5 = Conv(128,256,stride=2)
        self.context5 = ResBlock(256)
        self.up1 = UpSample(256,128)



    
    def forward(self, x):
        
        

        return 




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
        out = self.drop(out)

        return torch.sum(out, x)



        
class Conv(nn.Module):
    def __init__(self, in_plane, out_plane, stride):
        super().__init__()
        self.conv = nn.Sequential(
                nn.Conv3d(in_plane, out_plane, stride=stride, kernel_size=3),
                nn.InstanceNorm3d(out_plane),
                F.leaky_relu(negative_slope=10**-2)
        )

    def forward(self, x):
        return self.convolute(x)


class UpSample(nn.Module):
    def __init__(self, in_plane, out_plane):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2)
        self.conv = Conv(in_plane, out_plane, stride=1)

    def forward(self, x):
        out = self.up(x)
        out = self.conv(out)

        return out
        