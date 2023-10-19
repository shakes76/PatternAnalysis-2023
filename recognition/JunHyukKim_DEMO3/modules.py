import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torch
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
CUDA_DEVICE_NUM = 0
NUM_EPOCHS = 15
BATCH_SIZE = 1
WORKERS = 4
LEARNING_RATE = 0.0001
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ContextLayer(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(ContextLayer,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1,stride=1,bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Conv2d(out_channels, out_channels, 3, padding=1,stride=1,bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
    )
    def forward(self, x):
        return self.conv(x)
    
class ConvLayer(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(ConvLayer,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1,stride=2,bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)
    
class Localization(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(Localization,self).__init__()
        self.loc = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1,stride=1,bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1,stride=1,bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
    )
    def forward(self, x):
        return self.loc(x)
    
class Upsampling(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(Upsampling,self).__init__()
        self.upsampleing = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, 3, padding=1,stride=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        return self.upsampleing(x)
    
class UNet(nn.Module):
    def __init__(self, input_channels=3,out_channels=1,features=[64,128,256,512]):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(2,2)
        self.first_conv = nn.Conv2d(3, 64, 3, padding=1,stride=1)
        self.first_context = ContextLayer(64, 64)
        self.second_context = ContextLayer(128, 128)
        self.third_context = ContextLayer(256, 256)
        self.fourth_context = ContextLayer(512, 512)
        self.fifth_context = ContextLayer(512*2, 512*2)
        self.first_down = nn.Conv2d(64, 128, kernel_size=3,stride=2,padding=1)
        self.second_down = nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1)
        self.third_down = nn.Conv2d(256,512,kernel_size=3,stride=2,padding=1)
        self.fourth_down = nn.Conv2d(512,512*2,kernel_size=3,stride=2,padding=1)

        self.segment_upscale = nn.Upsample(scale_factor=2,mode='nearest')
        
        self.first_up = Upsampling(512*2,512)
        self.second_up = Upsampling(512,256)
        self.third_up = Upsampling(256,128)
        self.fourth_up = Upsampling(128,64)

        self.first_local = Localization(512*2,512)
        self.second_local = Localization(256*2,256)
        self.third_local = Localization(128*2,128)

        self.segment_layer1 = nn.Conv2d(256, 1, 1,stride=1)
        self.segment_layer2 = nn.Conv2d(128, 1, 1,stride=1)
        self.segment_layer3 = nn.Conv2d(128, 1, 1,stride=1)

        self.final_conv = nn.Conv2d(128,128,kernel_size=3,padding=1)
        self.final_segmentIDK = nn.Conv2d(128,1,kernel_size=1)
        self.final_activiation = nn.Sigmoid()    
    
    
    def forward(self, x):
        #LAYER 1
        x = self.first_conv(x)
        context1 = self.first_context(x)
        x.add(context1)
        skip_connection1 = x

        #LAYER 2
        x = self.first_down(x)
        context2 = self.second_context(x)
        x.add(context2)        
        skip_connection2 = x

        #LAYER 3
        x = self.second_down(x)
        context3 = self.third_context(x)
        x.add(context3)        
        skip_connection3 = x

        #LAYER 4
        x = self.third_down(x)
        context4 = self.fourth_context(x)
        x.add(context4)        
        skip_connection4 = x

        #LAYER 5
        x = self.fourth_down(x)
        context5 = self.fifth_context(x)
        x.add(context5)        
        x = self.first_up(x)
        
        #LAYER -4   
        x = torch.cat((skip_connection4, x), dim=1)
        x = self.first_local(x)
        x = self.second_up(x)

        #LAYER -3   
        x = torch.cat((skip_connection3, x), dim=1)
        x = self.second_local(x)
        segment1 = self.segment_layer1(x)
        segment1 = self.segment_upscale(segment1)
        x = self.third_up(x)

        #LAYER -2   
        x = torch.cat((skip_connection2, x), dim=1)
        x = self.third_local(x)
        segment2 = self.segment_layer2(x)   
        segment2.add(segment1)
        segment2 = self.segment_upscale(segment2)
        x = self.fourth_up(x)

        #LAYER -1
        concat_skip = torch.cat((skip_connection1, x), dim=1)
        x = self.final_conv(concat_skip)
        segment3 = self.segment_layer3(x) 
        segment3.add(segment2)
        return self.final_activiation(segment3)