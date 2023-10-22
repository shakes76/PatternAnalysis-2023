import torch
from torch import nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as funk


#Pre-activation and context block with two 3x3 convolutional layers with a drop out inbetween. 
class pre_act_context(nn.Module):
    
    def __init__(self,in_channels,out_channels):
        super(pre_act_context,self).__init__()
        self.pre_activation=nn.Sequential(
            convoltion(in_channels,out_channels,3,1,1), #Convolutional layer 
            nn.Dropout(p=0.3),                          #Dropout that zeros elements based on a probability
            convoltion(in_channels,out_channels,3,1,1)
        )

    def forward(self,x):
        out=self.pre_activation(x) #Runs the pre-activation 
        out=torch.add(out,x) #Adds the input and the outout of the pre-actiavtion by elementwise sum
        return out
    
#The convoluional layer which consists of a batch norm, a relu and a convolution
class convoltion(nn.Module):
    def __init__(self,in_channels,out_channels,kernel,stride,padding):
        super(convoltion,self).__init__()
        self.bn1=nn.BatchNorm2d(in_channels)                                                                        #Batch normalization
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=kernel,stride=stride,padding=padding,bias=False)  #Convolution

    def forward(self,x):
        out=self.bn1(x)    #Batch nomalization
        out=funk.relu(out) #Relu
        out=self.conv1(out)#Convolution
        return out

#Localization module which consists of a 3x3 convolution and one 1x1 convolution        
class local(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(local,self).__init__()
        self.localization=nn.Sequential(
            convoltion(in_channels,in_channels,3,1,1), #3x3 convolution
            convoltion(in_channels,out_channels,1,1,0) #1x1 convolatuion
        )

    def forward(self,x):
        out=self.localization(x)
        return out

#Down-sampling block that uses a 3x3 convoliutional layer with a stride of 2 for halving the spatial dimensions.
#This block leads directly into the pre activation block. 
class down_samp(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(down_samp,self).__init__()
        self.down_sampling=nn.Sequential(
            convoltion(in_channels,out_channels,3,2,1),    #3x3 convolution with stride 2
            pre_act_context(out_channels,out_channels)     #Calling the pre-activating and context block
        )

    def forward(self,x):
        out=self.down_sampling(x)
        return out
    
#Upscaling block consisting of a convolution and a interpolation that increases the spatila dimensions by 2
class up_scale(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(up_scale,self).__init__()
        self.up_scale=nn.Sequential(
            convoltion(in_channels,out_channels,3,1,1) #3x3 convolutional layer
        )
        
    def forward(self,x):
        out=F.interpolate(x,scale_factor=2,mode='nearest') #Interpolation with scale factor 2
        out=self.up_scale(out)                             #3x3 convolution
        return out

#Segmentation layer that uses a 1x1 convolution 
class segmentation(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(segmentation,self).__init__()
        self.segment=convoltion(in_channels,out_channels,1,1,0)  #1x1 convolution
    
    def forward(self,x):        
        out=self.segment(x)
        return out
        


class UNet(nn.Module):
    def __init__(self):
        super(UNet,self).__init__()
        self.conv1=convoltion(3,16,3,1,1)
        self.down1=pre_act_context(16, 16)

        self.down2=down_samp(16, 32)
        self.down3=down_samp(32, 64)
        self.down4=down_samp(64, 128)
        self.down5=down_samp(128, 256)

        self.up1=up_scale(256,128)
        self.local1=local(256,128)

        self.up2=up_scale(128,64)
        self.local2=local(128,64)
        self.segm1=segmentation(64,1)

        self.up3=up_scale(64,32)
        self.local3=local(64,32)
        self.segm2=segmentation(32,1)

        self.up4=up_scale(32,16)
        self.conv4=convoltion(32,32,3,1,1)
        self.segm3=segmentation(32,1)

        self.sigmoid=nn.Sigmoid()
    
    def forward(self,x):
        out=self.conv1(x)               #Initial convolution
        out=self.down1(out)             #Down sampling + preactivation + context
        cat4=out                        #Saving output to concatenate on the upscaling
        out=self.down2(out)
        cat3=out
        out=self.down3(out)
        cat2=out
        out=self.down4(out)
        cat1=out
        out=self.down5(out)
        
        out=self.up1(out)               #Upsampling
        out=torch.cat((out,cat1),dim=1) #Concatenate with corresponding level form contraction pathway
        out=self.local1(out)            #Localization module after concatenating  

        out=self.up2(out)
        out=torch.cat((out,cat2),dim=1)
        out=self.local2(out)
        segm1=self.segm1(out)                                     #Saving output for segmentation pathway allowing for deep supervision
        segm1=F.interpolate(segm1,scale_factor=2,mode='bilinear') #Rescalig the output match later levels in expansion pathway

        out=self.up3(out)
        out=torch.cat((out,cat3),dim=1)
        out=self.local3(out)
        segm2=self.segm2(out)
        segm2=torch.add(segm1,segm2)                                #Element wise sum for the segmentation pathway
        segm2=F.interpolate(segm1,scale_factor=2,mode='bilinear')   #Rescaling the output to match later levesl in expansion pathway

        out=self.up4(out)
        out=torch.cat((out,cat4),dim=1)
        out=self.conv4(out)
        segm3=self.segm3(out)
        segm3=torch.add(segm2,segm3)                               #Rejoining the segmentation pathway with the exspansion pathway by elementwise sum
        
        out=self.sigmoid(segm3)                                    #Using sigmoid to get probability values between 0 and 1
        return out



