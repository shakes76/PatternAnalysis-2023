import torch
import torch.nn as nn
import torch.nn.functional as F

# REF: DiceLoss function gotten from
# REF: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch?fbclid=IwAR3q7bjIDoKFlc5IDGpd24TW8QhQdzbxh2TrIP6FCXb7A8FaluU_HhTqmHA
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        
    def forward(self, predict, target):
        predict = predict.view(-1)
        target = target.view(-1)

        intersect = (predict * target).sum()
        dice = (2*intersect)/(predict.sum() + target.sum())
        return 1 - dice

class Context(nn.Module):
    def __init__(self, size):
        super(Context, self).__init__()
        self.pdrop = 0.3
        self.neagative_slope = 10**(-2)

        self.batchNorm = nn.BatchNorm2d(size)
        self.conv = nn.Conv2d(
            size, size, kernel_size=3, padding=(1,1)
        )
        self.dropOut = nn.Dropout2d(self.pdrop)

    def forward(self, input):
        out = input
        out = self.conv(F.relu(self.batchNorm(out)))
        #out = F.relu(self.batchNorm(self.conv(out)))
        out = self.dropOut(out)
        out = F.relu(self.batchNorm(self.conv(out)))
        return torch.add(out, input)

class Upsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsampling, self).__init__()

        self.upscale = nn.Upsample(scale_factor=2)
        self.batchNorm = nn.BatchNorm2d(out_channels)
        self.conv = nn.Conv2d(
           in_channels, out_channels, kernel_size=3 , padding=(1,1)
        ) 

    def forward(self, out):
        out = self.upscale(out)
        out = F.relu(self.batchNorm(self.conv(out)))
        return out
        
class Localization(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Localization, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=(1,1)
        )
        self.batchNorm1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1
        )
        self.batchNorm2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, out):
        out = F.relu(self.batchNorm1(self.conv1(out)))
        out = F.relu(self.batchNorm2(self.conv2(out))) 
        return out
        
class ImpUNet(nn.Module):
    def __init__(self, in_channel, negative_slope=10**(-2)):
        super(ImpUNet, self).__init__()
        self.negative_slope = negative_slope
        # arcitecture components
        self.conv1 = nn.Conv2d(
            in_channel, 16, kernel_size=3, padding=(1,1)
        )
        self.batchNorm1 = nn.BatchNorm2d(16)
        self.context1 = Context(16)

        self.conv2 = nn.Conv2d(
            16, 32, kernel_size=3, stride=2, padding=(1,1)
        )
        self.batchNorm2 = nn.BatchNorm2d(32)
        self.context2 = Context(32)

        self.conv3 = nn.Conv2d(
            32, 64, kernel_size=3, stride=2, padding=(1,1)
        )
        self.batchNorm3 = nn.BatchNorm2d(64)
        self.context3 = Context(64)
        
        self.conv4 = nn.Conv2d(
            64, 128, kernel_size=3, stride=2, padding=(1,1)
        )
        self.batchNorm4 = nn.BatchNorm2d(128)
        self.context4 = Context(128)
        
        self.conv5 = nn.Conv2d(
            128, 256, kernel_size=3, stride=2, padding=(1,1)
        )
        self.batchNorm5 = nn.BatchNorm2d(256)
        self.context5 = Context(256)
        
        self.upsample0 = Upsampling(256, 128)
        
        self.localize1 = Localization(256, 128)
        self.upsample1 = Upsampling(128, 64)

        self.localize2 = Localization(128, 64)
        self.upsample2 = Upsampling(64, 32)

        self.localize3 = Localization(64, 32)
        self.upsample3 = Upsampling(32, 16)
        
        self.conv6 = nn.Conv2d(
            32, 32, kernel_size=3, padding=(1,1)
        )
        self.batchNorm6 = nn.BatchNorm2d(32)

        self.upscale = nn.Upsample(scale_factor=2)
        
        self.Sigmoid = nn.Sigmoid()
        # segmentation layers
        self.seg1 = nn.Conv2d(
            64, 1, kernel_size=1
        )
        self.segNorm1 = nn.BatchNorm2d(1)
        self.seg2 = nn.Conv2d(
            32, 1, kernel_size=1
        )
        self.segNorm2 = nn.BatchNorm2d(1)
        self.seg3 = nn.Conv2d(
            32, 1, kernel_size=1
        )
        self.segNorm3 = nn.BatchNorm2d(1)
        
    def forward(self, out):
        # convolution layer. input 3d image output 16 channels
        out = F.relu(self.batchNorm1(self.conv1(out))) 
        # contaxt block. input/output 16 channels
        out = self.context1(out)
        # save information
        layer1 = out
        # convolution layer. input 16 channels, output 32 channels
        out = F.relu(self.batchNorm2(self.conv2(out)))
        # context block. input/output 32 channels
        out = self.context2(out)
        # save information
        layer2 = out
        # convolution layer. input 32 channels. output 64 channels
        out = F.relu(self.batchNorm3(self.conv3(out))) 
        # context block. input/output 64 channels
        out = self.context3(out)
        # save information
        layer3 = out
        # convolutional layer. input 64 channels. output 128 channels
        out = F.relu(self.batchNorm4(self.conv4(out)))
        # context block. input/output 128 channels
        out = self.context4(out)
        # save information
        layer4 = out
        # convolutional layer. input 128 channels. output 256 channels
        out = F.relu(self.batchNorm5(self.conv5(out)))
        # context block. input/output 256 channels
        out = self.context5(out)
        # upsample module. input 256 channels. output 128 channels
        out = self.upsample0(out)
        # concatinate. 0 here is dimension
        out = torch.cat((out, layer4), 1)
        # localization module. input 256 channels. output 128 channels
        out = self.localize1(out)
        # upsampling module. input 128 channels. output 64 channels
        out = self.upsample1(out)
        # concatinate 
        out = torch.cat((out, layer3), 1)
        # localization layer. input 128 channels. output 64 channels
        out = self.localize2(out)
        # first segmentation layer
        seg1 = self.upscale(F.relu(self.segNorm1(self.seg1(out))))
        # upsampling module. input 64 channels. output 32 channels
        out = self.upsample2(out)
        # concatinate
        out = torch.cat((out, layer2), 1)
        # localization module. input 64 channels, output 32 channels
        out = self.localize3(out)
        # second segmentation layer
        seg2 = self.upscale(torch.add(F.relu(self.segNorm2(self.seg2(out))), seg1))
        # upsampling module. input 32 channels. output 16 channels
        out = self.upsample3(out)
        # concatinate
        out = torch.cat((out, layer1), 1)
        # convolutional layer. input/output 32 channels
        out = F.relu(self.batchNorm6(self.conv6(out)))
        # elementwise summation of current out and seg2
        out = torch.add(F.relu(self.segNorm3(self.seg3(out))), seg2)
        # softmax
        out = self.Sigmoid(out)
        return out
