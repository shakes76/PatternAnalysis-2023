import torch
from torch import nn
from dataset_test import CustomDataset
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
        self.dropout=nn.Dropout(p=0.3)
        self.bn2=nn.BatchNorm2d(in_channels)
        self.conv2=nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)

    def forward(self,x):
        out=self.bn1(x)
        out=funk.relu(out)
        out=self.conv1(out)
        out=self.dropout(out)
        out=self.bn2(out)
        out=funk.relu(out)
        out=self.conv2(out)
        out=torch.add(out,x)
        self.output=out
        return out
    
class local(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.bn1=nn.BatchNorm2d(in_channels)
        self.conv1=nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2=nn.BatchNorm2d(in_channels)
        self.conv2=nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

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
        super().__init__()
        self.bn1=nn.BatchNorm2d(in_channels)
        self.context_down=nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=2,padding=1,bias=False), pre_act(out_channels,out_channels,name))

    def forward(self,x):
        out=self.bn1(x)
        out=funk.relu(out)
        out=self.context_down(out)
        return out


class up_scale(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.bn1=nn.BatchNorm2d(in_channels)
        self.up_samp=nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        
    
    def forward(self,x):
        out=F.interpolate(x,scale_factor=2,mode='bilinear')
        out=self.bn1(out)
        out=funk.relu(out) 
        out=self.up_samp(out)
        return out

class segmentation(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.bn1=nn.BatchNorm2d(in_channels)
        self.segm=nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False) 
    
    def forward(self,x):        
        out=self.bn1(x)
        out=funk.relu(out) 
        out=self.segm(out)
        return out
        


class UNet(nn.Module):
    def __init__(self):
        super(UNet,self).__init__()
        self.conv1=nn.Conv2d(3,16,stride=1,kernel_size=3,padding=1,bias=False)
        self.down1=pre_act(16, 16, "1_down")

        self.down2=down_samp(16, 32, "2_down")
        self.down3=down_samp(32, 64, "3_down")
        self.down4=down_samp(64, 128, "4_down")
        self.down5=down_samp(128, 256, "5_down")

        self.up1=up_scale(256,128)
        self.local1=local(256,128)

        self.up2=up_scale(128,64)
        self.local2=local(128,64)
        self.segm1=segmentation(64,1)

        self.up3=up_scale(64,32)
        self.local3=local(64,32)
        self.segm2=segmentation(32,1)

        self.up4=up_scale(32,16)
        self.conv4=nn.Conv2d(32,32,stride=1,kernel_size=3,padding=1,bias=False)
        self.segm3=segmentation(32,1)

        self.softmax=nn.Softmax(dim=3)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self,x):
        out=self.conv1(x)
        out=self.down1(out)
        cat4=out
        out=self.down2(out)
        cat3=out
        out=self.down3(out)
        cat2=out
        out=self.down4(out)
        cat1=out
        out=self.down5(out)
        
        out=self.up1(out)
        out=torch.cat((out,cat1),dim=1)
        out=self.local1(out)

        out=self.up2(out)
        out=torch.cat((out,cat2),dim=1)
        out=self.local2(out)
        segm1=self.segm1(out)
        segm1=F.interpolate(segm1,scale_factor=2,mode='bilinear')

        out=self.up3(out)
        out=torch.cat((out,cat3),dim=1)
        out=self.local3(out)
        segm2=self.segm2(out)
        segm2=torch.add(segm1,segm2)
        segm2=F.interpolate(segm1,scale_factor=2,mode='bilinear')

        out=self.up4(out)
        out=torch.cat((out,cat4),dim=1)
        out=self.conv4(out)
        segm3=self.segm3(out)
        segm3=torch.add(segm2,segm3)
        
        #softmax=self.softmax(segm3)
        sigmoid=self.sigmoid(segm3)
        return sigmoid




