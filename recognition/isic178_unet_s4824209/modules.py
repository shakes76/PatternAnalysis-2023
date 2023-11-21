'''
Author: 48242099

Program for creating the improved uNet model

'''

import torch
import torch.nn as nn

    
class IuNet(nn.Module):
    '''
        Improved uNEt class
        Structure explained in README.md and in https://arxiv.org/pdf/1802.10508v1.pdf
    '''
    def __init__(self):
        super(IuNet, self).__init__()

        #Encoder
        self.conv1 = Conv(3, 16, stride=1, kernel_size=3)
        self.context1 = Context(16)
        self.conv2 = Conv(16,32, stride=2, kernel_size=3)
        self.context2 = Context(32)
        self.conv3 = Conv(32,64, stride=2, kernel_size=3)
        self.context3 = Context(64)
        self.conv4 = Conv(64,128, stride=2, kernel_size=3)
        self.context4 = Context(128)
        self.conv5 = Conv(128,256,stride=2, kernel_size=3)
        self.context5 = Context(256)
        
        #Decoder
        self.up1 = UpSample(256,128)
        self.loc1 = localization(256, 128)
        self.up2 = UpSample(128,64)
        self.loc2 = localization(128,64)
        self.up3 = UpSample(64,32)
        self.loc3 = localization(64,32)
        self.up4 = UpSample(32,16)
        self.conv6 = Conv(32,32, stride=1, kernel_size=3)
        
        #segmentation layers
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.seg3conv = Conv(64,1, stride=1, kernel_size=1, padding=0)
        self.seg2conv = Conv(32,1, stride=1, kernel_size=1, padding=0)
        self.seg1conv = Conv(32,1, stride=1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #Layer 1 decoder // inputchannels:3 outputchannels:16
        lay1d = self.conv1(x)
        lay1d = self.context1(lay1d)
        
        #Layer 2 decoder // inputchannels: 16 outputchannels: 32
        lay2d = self.conv2(lay1d)
        lay2d = self.context2(lay2d)
        
        #Layer 3 decoder // inputchannels:32 outputchannels:64
        lay3d = self.conv3(lay2d)
        lay3d = self.context3(lay3d)
        
        #Layer 4 decoder // Inputchannels: 64, Outputchannels: 128
        lay4d = self.conv4(lay3d)
        lay4d = self.context4(lay4d)
        
        #Bottleneck // Inputchannels: 128, Outputchannels: 128
        lay5 = self.conv5(lay4d)
        lay5 = self.context5(lay5)
        lay5 = self.up1(lay5)
        
        #Layer 4 encoder // Inputchannels: 256, Outputchannels: 64
        lay4u = torch.concat((lay4d, lay5), dim=1)
        lay4u = self.loc1(lay4u)
        lay4u = self.up2(lay4u)

        #Layer 3 encoder // Inputchannels: 128, Outputchannels: 32 
        lay3u = torch.concat((lay3d, lay4u), dim=1)
        seg3 = self.loc2(lay3u)     #Output used for segmentation // Outputchannels:64
        lay3u = self.up3(seg3)

        #Layer 2 encoder // Input: 64, Output: 16
        lay2u = torch.concat((lay2d, lay3u), dim=1)
        seg2 = self.loc3(lay2u)     #Output used for segmentation // Outputchannels:32
        lay2u = self.up4(seg2)

        #Layer 1 encoder // Inputchannels: 32, Outputchannels: 32
        lay1u = torch.concat((lay1d, lay2u), dim=1)
        lay1u = self.conv6(lay1u)

        #Segmentation layers // Outputchannels: 1
        seg3 = self.seg3conv(seg3)
        seg2 = self.seg2conv(seg2)
        seg1 = self.seg1conv(lay1u)
        
        seg2 = torch.add(seg2, self.upsample(seg3))
        seg1 = torch.add(seg1, self.upsample(seg2))

        out = self.sigmoid(seg1)
        
        return out



class Context(nn.Module):
    '''
    Context module (pre-activation residual block)
    Consists of 2 (3x3) convolutional layers, with a drop layer (p=0.3) inbetween.
    Contains a skip-layer, so it returns the elementwise sum of input and output

    ***Note: the convolution layers has input channels = output channels

        returns: 
        
    '''
    def __init__(self, in_chnls):
        super(Context, self).__init__()
        self.context = nn.Sequential(
            nn.BatchNorm2d(in_chnls),
            nn.LeakyReLU(negative_slope=10**-2),
            nn.Conv2d(in_chnls, in_chnls, kernel_size=3, padding=(1,1)),
            nn.Dropout(p=0.3),
            nn.Conv2d(in_chnls, in_chnls, kernel_size=3, padding=(1,1)),
            nn.BatchNorm2d(in_chnls),
            nn.LeakyReLU(negative_slope=10**-2),
            )
        
    def forward(self, x):
        out = self.context(x)
        return torch.add(out, x)  


#Convolution  
class Conv(nn.Module):
    '''
    Convolution block with Batch-normalisation and leaky-relu

    args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channesls 
        stride (int): 
        kernel_size (int or tuple): Size of convolution kernel
        padding (int or tuple):

    returns: [x,out_channels,x,x] (tensor)
    '''

    def __init__(self, in_channels, out_channels, stride, kernel_size, padding = (1,1)):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(negative_slope=10**-2)
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class UpSample(nn.Module):
    '''
    Upsampling module with scale 2 upsample with interpolation mode nearest
    and a (3x3), stride=1, convolution with batchnorm and leakyReLU.

    args:
        in_channels: Number of input channels
        out_channels: Number of output channels 
    '''

    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = Conv(in_channels, out_channels, stride=1, kernel_size=3)

    def forward(self, x):
        out = self.up(x)
        out = self.conv(out)

        return out
        

class localization(nn.Module):
    '''
    Localization module with 2 convoluton layers (with batchnormalisation and ReLu) 

    args:
        in_channels(int): Number of input channels
        out_channels(int): Number of output channels
    '''
    def __init__(self, in_channels, out_channels):
        super(localization, self).__init__()
        self.conv1 = Conv(in_channels, in_channels, stride=1, kernel_size=3)
        self.conv2 = Conv(in_channels, out_channels, stride=1, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        
        return out


#create model:
model = IuNet()

