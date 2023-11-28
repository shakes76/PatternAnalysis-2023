
import torch
import torch.nn.functional as F
import torch.nn as nn


#Class implementation of model source code

class ImprovedUNET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImprovedUNET, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.context1 = context_module(out_channels, out_channels)
       
        #Reducing the number of channels to half for each layer
        in_channels = out_channels
        out_channels = out_channels * 2
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2)
        self.context2 = context_module(out_channels, out_channels)
        
        in_channels = in_channels * 2
        out_channels = out_channels * 2
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2)
        self.context3 = context_module(out_channels, out_channels)
        
        in_channels = in_channels * 2
        out_channels = out_channels * 2
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2)
        self.context4 = context_module(out_channels, out_channels)
        
        in_channels = in_channels * 2
        out_channels = out_channels * 2
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2)
        self.context5 = context_module(out_channels, out_channels)
        
        in_channels = out_channels
        out_channels = out_channels // 2
        self.up1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.localization1 = localization_module(in_channels, out_channels)
        
        in_channels = in_channels // 2
        out_channels = out_channels // 2
        self.up2 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.localization2 = localization_module(in_channels, out_channels)
        self.segmentation_layer2 = nn.Conv2d(out_channels, out_channels//2, kernel_size=1)
        
        in_channels = in_channels // 2
        out_channels = out_channels // 2
        self.up3 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.localization3 = localization_module(in_channels, out_channels)
        self.segmentation_layer3 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        
        in_channels = in_channels // 2
        out_channels = out_channels // 2
        
        self.up4 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.lastconv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.segmentation_layer4 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sample = nn.Upsample(scale_factor=2)
        self.featureReduc = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        features = []
        x = self.conv1(x)
        elem1 = x
        x = self.context1(x)
        x = x + elem1
        features.append(x)
        x = self.conv2(x)
        elem2 = x
        x = self.context2(x)
        x = x + elem2
        features.append(x)
        x = self.conv3(x)
        elem3 = x
        x = self.context3(x)
        x = x + elem3
        features.append(x)
        x = self.conv4(x)
        elem4 = x
        x = self.context4(x)
        x = x + elem4
        features.append(x)
        x = self.conv5(x)
        elem5 = x
        x = self.context5(x)
        x = x + elem5
        x = self.up1(x)
        x = torch.cat((x, features[-1]), dim=1)
        x = self.localization1(x)
        x = self.up2(x)
        x = torch.cat((x, features[-2]), dim=1)
        x = self.localization2(x)
        segmentation2 = self.sample(self.segmentation_layer2(x))
        x = self.up3(x)
        x = torch.cat((x, features[-3]), dim=1)
        x = self.localization3(x)
        segmentation3 = self.segmentation_layer3(x)
        x = self.up4(x)
        x = torch.cat((x, features[-4]), dim=1)
        x = self.lastconv(x)
        
        #Adding the residual connections
        
        segmentation4 = self.segmentation_layer4(x)
        combine1 = segmentation2 + segmentation3
        combine1 = self.sample(combine1)
        conbine2 = combine1 + segmentation4
        combineLast = self.featureReduc(conbine2)
        final = nn.Sigmoid()(combineLast)
        return final

def context_module(in_channels, out_channels):
        module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=0.3),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01)
        )
        return module

def localization_module(in_channels, out_channels):
        module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.LeakyReLU(0.01)
        )
        return module
    


