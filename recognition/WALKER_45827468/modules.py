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
        self.conv = nn.Conv2d(size // 2, size // 2, kernel_size=3, padding=1)
        
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
        self.in1 = nn.InstanceNorm2d(16)
        # context
        self.context1 = Context(16)
        # 3x3x3 conv stride 2   16->32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.in2 = nn.InstanceNorm2d(32)
        # context
        self.context2 = Context(32)
        # 3x3x3 conv stride 2   32->64
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.in3 = nn.InstanceNorm2d(64)
        # context
        self.context3 = Context(64)
        # 3x3x3 conv stride 2   64->128
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.in4 = nn.InstanceNorm2d(128)
        # context
        self.context4 = Context(128)
        # 3x3x3 conv stride 2   128->256
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.in5 = nn.InstanceNorm2d(256)
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
        
        # segmentation
        self.seg1 = nn.Conv2d(64, 1, kernel_size=1)
        self.seg2 = nn.Conv2d(32, 1, kernel_size=1)
        self.seg3 = nn.Conv2d(32, 1, kernel_size=1)
        self.segNorm = nn.InstanceNorm2d(3)
        self.segRelu = nn.LeakyReLU(3)
        self.upscale = nn.Upsample(scale_factor=2)
        
        # soft max
        self.softmax = nn.Softmax(dim=3)
        
         
    def forward(self, input):
        # layer 1
        print(input.size())
        out = self.in1(self.conv1(input))
        print("CONV1", out.size())
        out = self.context1(out)
        out1 = out                      # save output for concat
        print("CONTEXT1", out.size())
        # layer 2
        out = self.in2(self.conv2(out))
        print("CONV2", out.size())
        out = self.context2(out)
        out2 = out                      # save output for concat
        print("CONTEXT2", out.size())
        # layer 3
        out = self.in3(self.conv3(out))
        print("CONV3", out.size())
        out = self.context3(out)
        out3 = out                      # save output for concat
        print("CONTEXT3", out.size())
        # layer 4
        out = self.in4(self.conv4(out))
        print("CONV4", out.size())
        out = self.context4(out)
        out4 = out                      # save output for concat
        print("CONTEXT4", out.size())
        # layer 5
        out = self.in5(self.conv5(out))
        print("CONV5", out.size())
        out = self.context5(out)
        print("CONTEXT5", out.size())
        # layer 4
        out = self.upsamp1(out)
        print("UPSAMP1", out.size())
        out = torch.cat((out,out4), 1)
        print("CAT1", out.size())
        out = self.local1(out)
        print("LOCAL1", out.size())
        # layer 3
        out = self.upsamp2(out)
        print("UPSAMP2", out.size())
        out = torch.cat((out,out3), 1)
        print("CAT2", out.size())
        out = self.local2(out)
        print("LOCAL2", out.size())
        # seg
        seg1 = self.upscale(self.segRelu(self.segNorm(self.seg1(out))))
        print("SEG1", seg1.size())
        # layer 2
        out = self.upsamp3(out)
        print("UPSAMP3", out.size())
        out = torch.cat((out,out2), 1)
        print("CAT2", out.size())
        out = self.local3(out)
        print("LOCAL3", out.size())
        # seg
        seg2 = self.segRelu(self.segNorm(self.seg2(out)))
        print("SEG2", seg2.size())
        seg2 = torch.add(seg2, seg1)
        seg2 = self.upscale(seg2)
        print("SEG2 AGAIN", seg2.size())
        # layer 1
        out = self.upsamp4(out)
        print("UPSAMP4", out.size())
        out = torch.cat((out,out1), 1)
        print("CAT3", out.size())
        out = self.conv6(out)
        print("CONV6", out.size())
        # seg
        out = self.segRelu(self.segNorm(self.seg3(out)))    # seg3
        print("SEG3", out.size())
        out = torch.add(out, seg2)                          # combine seg layers
        print("SEGOUT", out.size())
        # softmax
        out = self.softmax(out)
        print("SOFTMAX", out.size())
        return out
    
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        
    def forward(self, pred, mask):
        union = torch.sum(pred) + torch.sum(mask) 
        intersect = torch.sum(pred * mask)
        # calculate dice coefficient, add 1 to avoid dividing by 0
        dice = (2.0 * intersect + 1.0) / (union + 1.0)
        return 1 - dice