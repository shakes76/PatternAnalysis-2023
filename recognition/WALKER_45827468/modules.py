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
        self.instNorm = nn.InstanceNorm2d(size // 2)
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
    def __init__(self, size):
        super(Localisation, self).__init__()
        
        self.conv1 = nn.Conv2d(size, size, kernel_size=3, padding=1)
        self.instNorm = nn.InstanceNorm2d(size)
        self.relu1 = nn.LeakyReLU(size)
        
        self.conv2 = nn.Conv2d(size, size // 2, kernel_size=1)
        self.relu2 = nn.LeakyReLU(size // 2)
        
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
        
        # 3x3x3 conv            3->16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        # context
        self.context1 = Context(16)
        # 3x3x3 conv stride 2   16->32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        # context
        self.context2 = Context(32)
        # 3x3x3 conv stride 2   32->64
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        # context
        self.context3 = Context(64)
        # 3x3x3 conv stride 2   64->128
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        # context
        self.context4 = Context(128)
        # 3x3x3 conv stride 2   128->256
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        # context
        self.context5 = Context(256)
        
        # upsample              256->128
        self.upsamp1 = UpSampling(256)
        # localisation
        self.local1 = Localisation(2*128)
        # upsample              128->64
        self.upsamp2 = UpSampling(128)
        # localisation
        self.local2 = Localisation(2*64)
        # upsample              64->32
        self.upsamp3 = UpSampling(64)
        # localisation
        self.local3 = Localisation(2*32)
        # upsample              32->16
        self.upsamp4 = UpSampling(32)
        # 3x3x3 conv
        self.conv6 = nn.Conv2d(2*16, 32, kernel_size=3, padding=1)
        # segmentation ???
        
         
    def forward(self, input):
        # layer 1
        out = self.bn1(self.conv1(input))
        out = self.context1(out)
        out1 = out                      # save output for concat
        
        # layer 2
        out = self.bn2(self.conv2(out))
        out = self.context2(out)
        out2 = out                      # save output for concat
        
        # layer 3
        out = self.bn3(self.conv3(out))
        out = self.context3(out)
        out3 = out                      # save output for concat
        
        # layer 4
        out = self.bn4(self.conv4(out))
        out = self.context4(out)
        out4 = out                      # save output for concat
        
        # layer 5
        out = self.bn5(self.conv5(out))
        out = self.context5(out)
        
        # layer 4
        out = self.upsamp1(out)
        out = torch.cat((out,out4), 1)
        out = self.local1(out)
        
        # layer 3
        out = self.upsamp2(out)
        out = torch.cat((out,out3), 1)
        out = self.local2(out)
        # seg ?
        
        # layer 2
        out = self.upsamp3(out)
        out = torch.cat((out,out2), 1)
        out = self.local3(out)
        # seg ?
        
        # layer 1
        out = self.upsamp4(out)
        out = torch.cat((out,out1), 1)
        out = self.conv6(out)
        # seg ?
        
        # combine seg
        
        # softmax
        
        return out