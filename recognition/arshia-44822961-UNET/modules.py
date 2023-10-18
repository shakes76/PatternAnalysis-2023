# This file contains the Improved Unet Model architecture as well as module building blocks 

# imports 
import torch
import torch.nn as nn
import torch.nn.functional as F

# Global variables 
NEGATIVE_SLOPE = 10**2
DROPOUT_PROB = 0.3 

"""

padding = kernel - stride/2 , padding = 1 
changed padding to 0 to retain
"""
class StandardConv(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 kernel_size=3, stride=1, padding=1):
        super(StandardConv, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, 
                                kernel_size=kernel_size, stride=stride, padding=padding)
        self.instance_norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.instance_norm(x)
        x = self.relu(x)
        return x

# Pre-activation residual block 
# two 3x3 layers and a drop out layer 
# described as pre-activation res block with 2 convs with drop out layer in between 
# entire feature mapping process using leaky relu as described by the paper. 
class ContextModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ContextModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.instance_norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE, inplace=True)
        self.dropout = nn.Dropout(DROPOUT_PROB)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.instance_norm2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu2 = nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.instance_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.instance_norm2(x)
        x = self.relu2(x)
        return x

"""
Padding = 3 - 2/ 2 = 0 
Padding = 0 


"""
class ConvWithStride(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 kernel_size=3, stride=2, padding=0):
        super(ConvWithStride, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE, inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        return out
    
# Update upsampling module code - don't need upsample and conv2d. 
# Replacing with ConvTranspose2d
class UpsamplingModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsamplingModule, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, 
                                                 kernel_size=2, stride=2, padding=0)
        self.conv = nn.Conv2d(out_channels, out_channels, 
                              kernel_size=3, stride=1, padding=1)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE, inplace=True)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.conv(x)
        x = self.relu(x)
        return x

    
# Localisation Module 
# Recombines features together 
class LocalisationModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LocalisationModule, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE, inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
        self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu2 = nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE, inplace=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        return x


# Segmentation Module
# reduces depth of feature maps to 1 
class SegmentationModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegmentationModule, self).__init__()
        self.segmentation = nn.Conv2d(in_channels, out_channels, 
                                      kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.segmentation(x)

class UpScaleModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpScaleModule, self).__init__()
        self.upscale = nn.ConvTranspose2d(in_channels, out_channels, 
                                          kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        return self.upscale(x)


class ImprovedUnet(nn.Module):
    def __init__(self):
        super(ImprovedUnet, self).__init__()
        in_channels = 3 # RGB colours have three input channels.
        out_channels = 16
        self.conv_1 = StandardConv(in_channels, out_channels)
        self.context_1 = ContextModule(in_channels, out_channels)
        
        in_channels = 16 
        out_channels = 32 
        self.conv_2 = ConvWithStride(in_channels, out_channels)
        self.context_layer_2 = ContextModule(in_channels, out_channels)

        in_channels = 32 
        out_channels = 64
        self.conv_3 = ConvWithStride(in_channels, out_channels)
        self.context_layer_3 = ContextModule(in_channels, out_channels)

        in_channels = 64 
        out_channels = 128
        self.conv_4 = ConvWithStride(in_channels, out_channels)
        self.context_layer_4 = ContextModule(in_channels, out_channels)

        in_channels = 128
        out_channels = 256 
        self.conv_5 = ConvWithStride(in_channels, out_channels)
        self.context_layer_5 = ContextModule(in_channels, out_channels)

        in_channels = 256
        out_channels = 128 
        self.upsample_layer_1 = UpsamplingModule(in_channels, out_channels)

        in_channels = 256 
        out_channels = 128
        self.localise_layer_1 = LocalisationModule(in_channels,out_channels)

        in_channels = 128
        out_channels = 64
        self.upsample_layer_2 = UpsamplingModule(in_channels, out_channels)
        
        in_channels = 128 
        out_channels = 64 
        self.localise_layer_2 = LocalisationModule(in_channels, out_channels)

        in_channels = 64
        out_channels = 32
        self.upsample_layer_3 = UpsamplingModule(in_channels, out_channels)

        in_channels = 64 
        out_channels = 32 
        self.localise_layer_3 = LocalisationModule(in_channels, out_channels)

        # fourth upsample layer 
        in_channels = 32 
        out_channels = 16 
        self.upsample_layer_4 = UpsamplingModule(in_channels, out_channels)
    
        in_channels = 32
        out_channels = 32 
        self.conv_output = StandardConv(in_channels, out_channels)
        
        # first segmentation layer 
        in_channels =  64
        out_channels = 1 
        self.segmentation_layer_1 = SegmentationModule(in_channels, out_channels)
        
        # second segmentation layer 
        in_channels = 32
        out_channels = 1 
        self.segmentation_layer_2 = SegmentationModule(in_channels, out_channels)

        # third segmentation layer 
        in_channels = 32
        out_channels = 1 
        self.segmentation_layer_3 = SegmentationModule(in_channels, out_channels)

        # upscaling layers 
        self.upscale_1 = UpScaleModule(out_channels, out_channels)
        self.upscale_2 = UpScaleModule(out_channels, out_channels)
        

    def forward(self, x):
        conv_out_1 = self.conv_1(x)
        context_out_1 = self.context_1(conv_out_1)
        element_sum_1 = conv_out_1 + context_out_1 

        # second term 
        conv_out_2 = self.conv_2(element_sum_1)
        context_out_2  = self.context_layer_2(conv_out_2)
        element_sum_2 = conv_out_2 + context_out_2

        # third downsample
        conv_out_3 = self.conv_3(element_sum_2)
        context_out_3 = self.context_layer_3(conv_out_3)
        element_sum_3 = conv_out_3 + context_out_3 

        conv_out_4 = self.conv_4(element_sum_3)
        context_out_4 = self.context_layer_4(conv_out_4)
        element_sum_4 = conv_out_4 + context_out_4 

        conv_out_5 = self.conv_5(element_sum_4)
        context_out_5 = self.context_layer_5(conv_out_5)
        element_sum_5 = conv_out_5 + context_out_5 

        # First upsampling module. 
        upsample_out_1 = self.upsample_layer_1(element_sum_5)
        concat_1 = torch.cat((element_sum_4, upsample_out_1), dim=1)
 
        localisation_out_1 = self.localise_layer_1(concat_1)
        upsample_out_2 = self.upsample_layer_2(localisation_out_1)
        concat_2 = torch.cat((element_sum_3, upsample_out_2))

        localisation_out_2 = self.localise_layer_2(concat_2)
        upsample_out_3 = self.upsample_layer_3(localisation_out_2)
        concat_3 = torch.cat((element_sum_2, upsample_out_3))

        localisation_out_3 = self.localise_layer_3(concat_3)
        upsample_out_4 = self.upsample_layer_4(localisation_out_3)
        concat_4 = torch.cat((element_sum_1, upsample_out_4))
        
        segment_out_1 = self.segmentation_layer_1(localisation_out_2)
        upscale_out_1 = self.upscale_1(segment_out_1)

        segment_out_2 = self.segmentation_layer_2(localisation_out_3)
        seg_sum_1 = upscale_out_1 + segment_out_2

        upscale_out_2 = self.upscale_2(seg_sum_1)

        segment_out_1 = self.segmentation_layer_1(localisation_out_2)
        upscale_out_1 = self.upscale_1(segment_out_1)

        convoutput_out = self.conv_output(concat_4)
        segment_out_3 = self.segmentation_layer_3(convoutput_out)

        final_sum = upscale_out_2 + segment_out_3
        
        return torch.sigmoid(final_sum)


        
