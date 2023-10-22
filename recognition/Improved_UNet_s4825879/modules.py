import torch
import torch.nn as nn
import torch.nn.functional as F

# REF: Dice function gotten from
# REF: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch?fbclid=IwAR3q7bjIDoKFlc5IDGpd24TW8QhQdzbxh2TrIP6FCXb7A8FaluU_HhTqmHA
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, predict, target):
        # flatten tensors
        predict = predict.view(-1)
        target = target.view(-1)

        # calculate the intersect value
        intersect = (predict * target).sum()
        # compute dice score
        dice = (2.*intersect + self.smooth)/(predict.sum() + target.sum() + self.smooth)

        return 1 - dice

# Class for single convolution followed by batchnorm and leakyReLU
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv, self).__init__()
        negative_slope = 5*10**(-5)

        # sequential layer containing modules
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope)
        )

    def forward(self, out):
        return self.conv(out)

# context block containing two convolutional layers with a dropout layer inbetween
class Context(nn.Module):
    def __init__(self, size):
        super(Context, self).__init__()
        self.pdrop = 0.3
        self.neagative_slope = 10**(-2)

        # sequential layer containing modules
        self.context = nn.Sequential(
            Conv(size, size),
            nn.Dropout2d(self.pdrop),
            Conv(size, size)
        )
            
    def forward(self, input):
        out = input
        out = self.context(out)
        return torch.add(out, input)

# Upsampling block containing upsample layer followed by convolution
class Upsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsampling, self).__init__()

        # sequential layer containing modules
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv(in_channels, out_channels)
        )

    def forward(self, out):
        return self.upsample(out)
        
# localization block for recombining features after skip connections
class Localization(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Localization, self).__init__()

        # sequential layer containing modules
        self.localize = nn.Sequential(
            Conv(in_channels, in_channels, kernel_size=3),
            Conv(in_channels, out_channels, kernel_size=1, padding=0),
        )
        
    def forward(self, out):
        return self.localize(out)

class ImpUNet(nn.Module):
    def __init__(self, in_channels=3):
        super(ImpUNet, self).__init__()
        
        # ENCODING
        # layer 1
        self.conv1 = Conv(in_channels, 16)
        self.context1 = Context(16)
        
        # layer 2
        self.conv2 = Conv(16, 32, stride=2)
        self.context2 = Context(32)
        
        # layer 3
        self.conv3 = Conv(32, 64, stride=2)
        self.context3 = Context(64)
        
        # layer 4
        self.conv4 = Conv(64, 128, stride=2)
        self.context4 = Context(128)
        
        # BOTTLENECK 
        self.conv5 = Conv(128, 256, stride=2)
        self.context5 = Context(256)
        self.upsample = Upsampling(256, 128)
        
        # DECODING
        # layer 4
        self.localize1 = Localization(256, 128)
        self.upsample1 = Upsampling(128, 64)
        
        # layer 3
        self.localize2 = Localization(128, 64)
        self.upsample2 = Upsampling(64, 32)
        
        # layer 2
        self.localize3 = Localization(64, 32)
        self.upsample3 = Upsampling(32, 16)
        
        # layer 1
        self.conv6 = Conv(32, 32)
        
        # segmentation
        self.upscale = nn.Upsample(scale_factor=2)
        self.seg_conv1 = Conv(64, 1, kernel_size=1, padding=0)
        self.seg_conv2 = Conv(32, 1, kernel_size=1, padding=0)
               
        # final output
        self.sigmoid = nn.Sigmoid()

    def forward(self, out):
        # layer 1
        out = self.conv1(out)
        out = self.context1(out)
        skip_connection1 = out
        # layer 2
        out = self.conv2(out)
        out = self.context2(out)
        skip_connection2 = out
        # layer 3
        out = self.conv3(out)
        out = self.context3(out)
        skip_connection3 = out
        # layer 4
        out = self.conv4(out)
        out = self.context4(out)
        skip_connection4 = out
        # bottleneck
        out = self.conv5(out)
        out = self.context5(out)
        out = self.upsample(out)
        # layer 4
        out = torch.cat((out, skip_connection4), dim=1)
        out = self.localize1(out)
        out = self.upsample1(out)
        # layer 3
        out = torch.cat((out, skip_connection3), dim=1)
        out = self.localize2(out)
        seg1 = self.upscale(self.seg_conv1(out))
        out = self.upsample2(out)
        # layer 2
        out = torch.cat((out, skip_connection2), dim=1)
        out = self.localize3(out)
        seg2 = self.upscale(torch.add(self.seg_conv2(out), seg1))
        out = self.upsample3(out)
        # layer 1
        out = torch.cat((out, skip_connection1), dim=1)
        out = self.conv6(out)
        out = torch.add(self.seg_conv2(out), seg2)
        
        # sigmoid
        out = self.sigmoid(out)
        
        return out

model = ImpUNet(3)
