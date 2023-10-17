import torch
import torch.nn as nn
import torch.nn.functional as F

'''
    pre-activation residual block
    two 3x3x3 conv with dropout layer inbetween
    in_channels matches out_channels, constant size for convolution
'''
class Context(nn.Module):
    def __init__(self, size):
        super(Context, self).__init__()
        
        self.pdrop = 0.3    # from paper
        
        self.instNorm = nn.InstanceNorm2d(size)
        self.conv = nn.Conv2d(size, size, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(self.pdrop)
        self.relu = nn.LeakyReLU(size)
        
    def forward(self, input):
        out = self.relu(self.instNorm(self.conv(input)))
        out = self.dropout(out)
        out = self.relu(self.instNorm(self.conv(out)))
        out = torch.add(out, input)
        
        return out

'''
    upsample
    simple upscale that repeats feature voxels twice in each dimension,
     then 3x3x3 conv to halve num of feature maps 
''' 
class UpSampling(nn.Module):
    def __init__(self, size):
        super(UpSampling, self).__init__()
        self.upsamp = nn.Upsample(scale_factor=2)
        self.instNorm = nn.InstanceNorm2d(size)
        self.conv = nn.Conv2d(size, size // 2, kernel_size=3, padding=1)
        
    def forward(self, input):
        out = self.instNorm(self.upsamp(input))
        out = self.conv(out)
        
        return out

'''
    recombines upsampled features with concatenated features from 
     context aggregation pathway
    3x3x3 conv, followed by 1x1x1 conv
'''    
class Localisation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Localisation, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.instNorm = nn.InstanceNorm2d(out_channels)
        self.relu1 = nn.LeakyReLU(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels // 2, kernel_size=1)
        self.relu2 = nn.LeakyReLU(out_channels // 2)
        
    def forward(self, input):
        out = self.relu1(self.instNorm(self.conv1(input)))
        out = self.relu2(self.instNorm(self.conv2(out)))
        
        return out
    
'''
    improved UNet block
    implemented as per paper
'''
class ImprovedUNet(nn.Module):
    def __init__(self):
        super(ImprovedUNet, self).__init__()
        
        # 3x3x3 conv            16
        
        # context
        
        # 3x3x3 conv stride 2   32
        
        # context
        
        # 3x3x3 conv stride 2   64
        
        # context
        
        # 3x3x3 conv stride 2   128
        
        # context
        
        # 3x3x3 conv stride 2   256
        
        # upsample              128
        
        # concat
        
        # localisation
        
        # upsample              64
        
        # concat
        
        # localisation
        
        # upsample              32
        
        # concat
        
        # localisation
        
        # upsample              16
        
        # 3x3x3 conv
        
        # segmentation ???
        
        
        
    def forward(self, input):
        out = 0
        
        return out