import torch
from torch import nn
from dataset import CustomDataset
import torch.nn.functional as F
from torchvision.utils import save_image
import torch
import torch.nn as nn
import torch.nn.functional as funk



class pre_act(nn.Module):
    expansion=1
    
    def __init__(self,in_channels,out_channels,name):
        super(pre_act,self).__init__()
        self.output=0
        self.name=name
        self.bn1=nn.BatchNorm2d(in_channels)
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d()
        self.conv2=nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)

    def forward(self,x):
        out=self.bn1(x)
        out=funk.relu(out)
        out=self.conv1(out)
        out=nn.Dropout(p=0.3)
        out=self.bn2(out)
        out=funk.relu(out)
        out=self.conv2(out)
        out=torch.add(out,x)
        self.output=out
        return out
    
class local(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(local,self).__init__()
        self.bn1=nn.BatchNorm2d(in_channels)
        self.conv1=nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2=nn.BatchNorm2d(in_channels)
        self.conv2=nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=1, bias=False)

    def forward(self,x):
        out=self.bn1(x)
        out=funk.relu(out)
        out=self.conv1(out)
        out=self.bn2(out)
        out=funk.relu(out)
        out=self.conv2(out)
        return out


class down_samp(nn.Module):
    def __init__(self,in_channels,out_channels,name):
        super.__init__()
        self.context_down=nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=2,padding=1,bias=False), pre_act(out_channels,out_channels,name))

    def forward(self,x):
        out=self.context_down(x)
        return out


class up_scale(nn.Module):
    def __init__(self,x,in_channels,out_channels,cat_tensor):
        super.__init__()
        self.up_samp=nn.Sequential(F.interpolate(x,scale_factor=2,mode='nearest'), nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False))
        
    
    def forward(self,x,cat_tensor):
        out=self.up_samp(x)
        out=torch.cat((out,cat_tensor))
        return out

class segmentation(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.segm=nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=1, bias=False),F.interpolate(x,scale_factor=2,mode='bilinear'))
        


class UNet(nn.Module):
    def __init__(self,block,num_blocks,num_classes=10):
        super(UNet,self).__init__()
        self.conv1=nn.Conv2d(3,16,stride=1,padding=1,bias=False)
        self.down1=pre_act(16, 16, "1_down")

        self.down2=down_samp(16, 32, "2_down")
        self.down3=down_samp(32, 64, "3_down")
        self.down4=down_samp(64, 128, "4_down")
        self.down5=down_samp(128, 256, "5_down")

        self.up1=up_scale(256,128,self.down4)
        self.local1=local(256,128)

        self.up2=up_scale(128,64,self.down3)
        self.local2=local(128,64)
        self.segm1=segmentation(64,32)

        self.up3=up_scale(64,32,self.down2)
        self.local3=local(64,32)
        self.segm2=segmentation(32,16)

        self.up4=up_scale(32,16,self.down1)
        self.conv4=nn.Conv2d(32,32,stride=1,padding=1,bias=False)
        self.segm3=segmentation(32,16)
    
    def forward(self,x):
        out=self.conv1(x)
        out=self.down1(out)
        out=self.down2(out)
        out=self.down3(out)
        out=self.down4(out)
        out=self.down5(out)
        out=self.up1(out)
        out=self.local1(out)
        out=self.up2(out)
        out=self.local2(out)
        segm1=self.segm1(out)
        out=self.up3(out)
        out=self.local3(out)
        segm2=self.segm2(out)
        segm2=torch.add(segm1,segm2)
        out=self.up4(out)
        out=self.conv4(out)
        segm3=self.segm3(out)
        segm3=torch.add(segm2,segm3)
        return segm3




