# Modules.py contains modules necessary for building the model. 

import torch
import torch.nn as nn


# Building blocks of the unet. 
class StandardConv(nn.Module):
    # inherits base class from pytorch modules 
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(StandardConv, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.relu(x)
        return x
    

# Pre-activation residual block 
# two 3x3 layers and a drop out layer 
# described as pre-activation res block with 2 convs with drop out layer in between 
# entire feature mapping process using leaky relu as described by the paper. 

# TODO - batch normalisation is incorrect 
# The paper uses instance norm
class ContextModule(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.3):
        super(ContextModule, self).__init__()

        # Batch normalisation before ReLU
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.LeakyReLU(inplace=True)

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        
        # Dropout layer in between 
        self.dropout = nn.Dropout2d(p=dropout_p)

        # Batch normalisation before ReLU
        self.bn2 = nn.BatchNorm2d(out_channels)

        # RELU 
        self.relu2 = nn.LeakyReLU(out_channels)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        
    def forward(self, x):
        # Forward pass through the context module
        # Batch normalisation and ReLU before the first convolution
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        
        return out

# added second conv block with stride 2 
# MESSY??? combine with the other conv block when cleaning up. 
class Conv2dStride2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(Conv2dStride2, self).__init__()

        # Define a 3x3 convolutional layer with stride 2 and optional padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding)

        # Leaky RELU 
        self.relu = nn.LeakyReLU(out_channels)

    def forward(self, x):
        # Forward pass through the convolutional layer
        out = self.conv(x)
        out = self.relu(out)
        return out
    
# Upsampling Module 
# As per paper 
class UpsamplingModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsamplingModule, self).__init__()

        # Upsampling layer (repeat each feature voxel twice in each spatial dimension)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # 3x3 convolutional layer that halves the number of feature maps
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        #Leakly Relu 
        self.relu = nn.LeakyReLU(out_channels)

    def forward(self, x):
        upsampled_x = self.upsample(x)
        out = self.conv(upsampled_x)
        out = self.relu(out)
        return out
    
# Localisation Module 
# Recombines features together 
class LocalisationModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LocalisationModule, self).__init__()
        # 3x3 convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.LeakyReLU(out_channels)
        # 1x1 convolutional layer that halves the number of feature maps
        self.conv2 = nn.Conv2d(out_channels, out_channels // 2, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.LeakyReLU(out_channels//2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(x)
        out = self.conv2(out)
        out = self.relu2(x)
        return out


# Segmentation Module
# this seems to be just storage, and that is added up later.
# used to capture and store feature maps at different levels of abstraction 
# Storage class - not sure if this is correct yet. 
class SegmentationLayer(nn.Module):
    def __init__(self):
        super(SegmentationLayer, self).__init__()

    def forward(self, x):
        # This segmentation layer does not perform any operation,
        # it simply passes the input feature map as-is.
        return x


# Softmax 
# Place holder - this won't be actually required as its own class
# I'll just add it into the network at the very end since its just a softmax layer 
class SoftMaxLayer(nn.Module):
    def __init__(self, num_classes):
        super(SoftMaxLayer, self).__init__()
        # ADD IN THE FINAL UNET MODEL. - 
        self.softmax = nn.Softmax(dim=1)  # Softmax along dimension 1 (usually used for classification)

    def forward(self, x):
        out = self.softmax(x)
        return out

class ImprovedUnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        # write this entire thing and then go get help. 
        super(ImprovedUnet, self).__init__()

        # because RGB colour images are used. 
        in_channels = 3
        out_channels = 16
        self.conv_layer_1 = StandardConv(in_channels, out_channels,
                                         kernel_size=3, stride=1, padding=0 )
        self.context_layer_1 = ContextModule(in_channels,out_channels,dropout_p=0.3)

        in_channels = 16 
        out_channels = 32 
        # so these features are allowing for more features as we go down. 
        self.conv_stride_layer_1 = Conv2dStride2(in_channels, out_channels,kernel_size=3, padding=1)
        
        # confirm if input 16 or 32? 
        self.context_layer_2 = ContextModule(in_channels,out_channels, dropout_p=0.3)

        # input size changes here?
        in_channels = 32 
        out_channels = 64
        self.conv_stride_layer_2 = Conv2dStride2(in_channels, out_channels,kernel_size=3, padding=1)
        self.context_layer_3 = ContextModule(in_channels,out_channels, dropout_p=0.3)

        # now next layer down 
        in_channels = 64 
        out_channels = 128
        self.conv_stride_layer_3 = Conv2dStride2(in_channels, out_channels, kernel_size=3, padding=1)
        self.context_layer_4 = ContextModule(in_channels,out_channels)

        # and then next layer down again 
        in_channels = 128
        out_channels = 256 
        self.conv_stride_layer_4 = Conv2dStride2(in_channels, out_channels)
        self.context_layer_5 = ContextModule(in_channels, out_channels)

        # upsample module here.
        # input 128 and output is 256 
        in_channels = 256
        out_channels = 128 
        # yes - upsampling halves the number of feature maps. 
        self.upsample_layer_1 = UpsamplingModule(in_channels, out_channels)

        # upscaling part of the image 
        in_channels = 256 
        out_channels = 128
        self.localise_layer_1 = LocalisationModule(in_channels,out_channels)

        in_channels = 128
        out_channels = 64
        self.upsample_layer_2 = UpsamplingModule(in_channels, out_channels)

        in_channels = 64 
        out_channels = 32 
        self.localise_layer_2 = LocalisationModule(in_channels, out_channels)

        in_channels = 32
        out_channels = 16
        self.upsample_layer_3 = UpsamplingModule(in_channels, out_channels)

        in_channels = 16
        out_channels = 32 
        self.conv_layer_2 = StandardConv(in_channels, out_channels)

        # softmax layer 
       






    # TODO - fix naming 
    # VERIFY THAT CALLING THE FORWARD FUNCTION HERE MAKES SENSE 
    def forward(self, x):
        # downsampling is a repeat of conv layer and context module
        # and then their element wise addition here. 
        input_conv_out = self.conv_layer_1.forward(x)
        context_out_1 = self.context_layer_1.forward(input)
        # element sum into the next conv 
        element_sum_1= input_conv_out + context_out_1 

        # into first 3x3 stride 2 conv 
        conv3_layer1 = self.conv_stride_layer_1.forward(element_sum_1)
        context_out_2  = self.context_layer_2.forward(conv3_layer1)
        element_sum_2 = conv3_layer1 + context_out_2

        # input into the next stride 2 block 
        conv_out = self.conv_stride_layer_2.forward(element_sum_2)
        context_out_3 = self.context_layer_3.forward(conv_out)
        element_sum_3 = conv_out + context_out_3 

        # next downsample 
        conv_out_2 = self.conv_stride_layer_4.forward(element_sum_3)
        context_out_4 = self.context_layer_4.forward(conv_out_2)
        element_sum_4 = conv_out_2 + context_out_4 

        # First upsampling module. 
        upsample_out_1 = self.upsample_layer_1.forward(element_sum_4)

        # CONCAT with element sum 3 
        concat_1 = torch.cat((element_sum_4, upsample_out_1), dim=1)

        # add localisation module 
        localisation_out_1 = self.localise_layer_1.forward(concat_1)
        # fed directly into the upsampling layer 
        upsample_out_2 = self.upsample_layer_2.foward(localisation_out_1)

        # concat with element sum 3 here. 
        concat_2 = torch.cat((element_sum_3, upsample_out_2))

        # feed into localisation 2
        localisation_out_2 = self.localise_layer_1.forward(concat_1)




        


        pass 


        
