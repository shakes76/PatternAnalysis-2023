"""
This file implements an Improved UNet model to be used in train.py

NOTES:
- Dropout layers are not reused and instead defined separately for each layer that uses it
    - This is because of the concern that reusing the same layer may result in the same tensors
        being dropped across different layers
        - This has not been tested
- Instance normalisation and LeakyRelu was used instead of batch normalisation and relu as described in the paper
"""

import torch
from torch import nn
from torch.nn import functional as F

# Model
# uses as reference https://github.com/pykao/Modified-3D-UNet-Pytorch
class ImprovedUNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_n_filter = 4):
        super(ImprovedUNet, self).__init__()
        # Set up model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_n_filter = base_n_filter

        self.upscale = nn.Upsample(scale_factor=2, mode='nearest') #to be reused in forward pass

        # Level 1 context Layer
        self.conv3E11 = nn.Conv2d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)

        #Context Module
        self.conv3d2 = nn.Conv2d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropoutE1 = nn.Dropout2d(0.5)
        self.conv3d3 = nn.Conv2d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
        self.inorm3d = nn.InstanceNorm2d(self.base_n_filter)

        # Level 2 context
        self.conv3E21 = nn.Conv2d(self.base_n_filter, self.base_n_filter*2, kernel_size=3, stride=2, padding=1, bias=False)
        self.inorm3E21 = nn.InstanceNorm2d(self.base_n_filter*2)

        ##Context Module
        self.conv3E22 = nn.Conv2d(self.base_n_filter*2, self.base_n_filter*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropoutE2 = nn.Dropout2d(0.5)
        self.inorm3E22 = nn.InstanceNorm2d(self.base_n_filter*2)
        self.conv3E23 = nn.Conv2d(self.base_n_filter*2, self.base_n_filter*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.inorm3E23 = nn.InstanceNorm2d(self.base_n_filter*2)

        # Level 3 context
        self.conv3E31 = nn.Conv2d(self.base_n_filter*2, self.base_n_filter*4, kernel_size=3, stride=2, padding=1, bias=False)
        self.inorm3E31 = nn.InstanceNorm2d(self.base_n_filter*4)

        #Context Module
        self.conv3E32 = nn.Conv2d(self.base_n_filter*4, self.base_n_filter*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropoutE3 = nn.Dropout2d(0.5)
        self.inorm3E32 = nn.InstanceNorm2d(self.base_n_filter*4)
        self.conv3E33 = nn.Conv2d(self.base_n_filter*4, self.base_n_filter*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.inorm3E33 = nn.InstanceNorm2d(self.base_n_filter*4)

        # Level 4 context
        self.conv3E41 = nn.Conv2d(self.base_n_filter*4, self.base_n_filter*8, kernel_size=3, stride=2, padding=1, bias=False)
        self.inorm3E41 = nn.InstanceNorm2d(self.base_n_filter*8)

        #Context Module
        self.conv3E42 = nn.Conv2d(self.base_n_filter*8, self.base_n_filter*8, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropoutE4 = nn.Dropout2d(0.5)
        self.inorm3E42 = nn.InstanceNorm2d(self.base_n_filter*8)
        self.conv3E43 = nn.Conv2d(self.base_n_filter*8, self.base_n_filter*8, kernel_size=3, stride=1, padding=1, bias=False)
        self.inorm3E43 = nn.InstanceNorm2d(self.base_n_filter*8)

        # Level 5 context
        self.conv3E51 = nn.Conv2d(self.base_n_filter*8, self.base_n_filter*16, kernel_size=3, stride=2, padding=1, bias=False)
        self.inorm3E51 = nn.InstanceNorm2d(self.base_n_filter*16)

        #Context Module
        self.conv3E52 = nn.Conv2d(self.base_n_filter*16, self.base_n_filter*16, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropoutE5 = nn.Dropout2d(0.5)
        self.inorm3E52 = nn.InstanceNorm2d(self.base_n_filter*16)
        self.conv3E53 = nn.Conv2d(self.base_n_filter*16, self.base_n_filter*16, kernel_size=3, stride=1, padding=1, bias=False)
        self.inorm3E53 = nn.InstanceNorm2d(self.base_n_filter*16)

        # Level 0 local
        self.convD01 = nn.Conv2d(self.base_n_filter*16, self.base_n_filter*8, kernel_size=3, stride=1, padding=1, bias=False)
        self.inorm3D01 = nn.InstanceNorm2d(self.base_n_filter*8)

        # Level 1 local
        #Localisation Module
        self.convD11 = nn.Conv2d(self.base_n_filter*16, self.base_n_filter*16, kernel_size=3, stride=1, padding=1, bias=False)
        self.inorm3D11 = nn.InstanceNorm2d(self.base_n_filter*16)
        self.convD12 = nn.Conv2d(self.base_n_filter*16, self.base_n_filter*8, kernel_size=1, stride=1, padding=0, bias=False)
        self.inorm3D12 = nn.InstanceNorm2d(self.base_n_filter*8)

        #Upsample Module
        self.convD13 = nn.Conv2d(self.base_n_filter*8, self.base_n_filter*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.inorm3D13 = nn.InstanceNorm2d(self.base_n_filter*4)

        # Level 2 local
        #Localisation Module
        self.convD21 = nn.Conv2d(self.base_n_filter*8, self.base_n_filter*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.inorm3D21 = nn.InstanceNorm2d(self.base_n_filter*4)
        self.convD22 = nn.Conv2d(self.base_n_filter*4, self.base_n_filter*4, kernel_size=1, stride=1, padding=0, bias=False)
        self.inorm3D22 = nn.InstanceNorm2d(self.base_n_filter*4)

        #Upsample Module
        self.convD23 = nn.Conv2d(self.base_n_filter*4, self.base_n_filter*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.inorm3D23 = nn.InstanceNorm2d(self.base_n_filter*2)

        #Segmentation layer
        self.seg2 = nn.Conv2d(self.base_n_filter*4, 1, kernel_size=3, stride=1, padding=1, bias=False)

        # Level 3 local
        #Localisation Module
        self.convD31 = nn.Conv2d(self.base_n_filter*4, self.base_n_filter*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.inorm3D31 = nn.InstanceNorm2d(self.base_n_filter*2)
        self.convD32 = nn.Conv2d(self.base_n_filter*2, self.base_n_filter*2, kernel_size=1, stride=1, padding=0, bias=False)
        self.inorm3D32 = nn.InstanceNorm2d(self.base_n_filter*2)

        #Upsample Module
        self.convD33 = nn.Conv2d(self.base_n_filter*2, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
        self.inorm3D33 = nn.InstanceNorm2d(self.base_n_filter)

        #Segmentation layer
        self.seg3 = nn.Conv2d(self.base_n_filter*2, 1, kernel_size=3, stride=1, padding=1, bias=False)

        # Level 4 localization layer
        self.conv3D41 = nn.Conv2d(self.base_n_filter*2, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
        self.inorm3D4 = nn.InstanceNorm2d(self.base_n_filter)

        # Output layer
        self.convOut = nn.Conv2d(self.base_n_filter, self.out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #  Level 1 context layer
        out = self.conv3E11(x)
        sum1 = out
        out = F.leaky_relu(out)
        out = self.conv3d2(out)
        out = self.dropoutE1(out)
        out = F.leaky_relu(out)
        out = self.conv3d3(out)
        out = torch.add(out, sum1)
        skip1 = F.leaky_relu(out)
        out = self.inorm3d(out)
        out = F.leaky_relu(out)

        # Level 2 context layer
        out = self.conv3E21(out)
        sum2 = out
        out = self.inorm3E21(out)
        out = F.leaky_relu(out)
        out = self.conv3E22(out)
        out = self.dropoutE2(out)
        out = self.inorm3E22(out)
        out = F.leaky_relu(out)
        out = self.conv3E23(out)
        out = torch.add(out, sum2)
        skip2 = F.leaky_relu(out)
        out = self.inorm3E23(out)
        out = F.leaky_relu(out)

        # Level 3 context layer
        out = self.conv3E31(out)
        sum3 = out
        out = self.inorm3E31(out)
        out = F.leaky_relu(out)
        out = self.conv3E32(out)
        out = self.dropoutE3(out)
        out = self.inorm3E32(out)
        out = F.leaky_relu(out)
        out = self.conv3E33(out)
        out = torch.add(out, sum3)
        skip3 = F.leaky_relu(out)
        out = self.inorm3E33(out)
        out = F.leaky_relu(out)

        # Level 4 context layer
        out = self.conv3E41(out)
        sum4 = out
        out = self.inorm3E41(out)
        out = F.leaky_relu(out)
        out = self.conv3E42(out)
        out = self.dropoutE4(out)
        out = self.inorm3E42(out)
        out = F.leaky_relu(out)
        out = self.conv3E43(out)
        out = torch.add(out, sum4)
        skip4 = F.leaky_relu(out)
        out = self.inorm3E43(out)
        out = F.leaky_relu(out)

        # Level 5 context layer
        out = self.conv3E51(out)
        sum5 = out
        out = self.inorm3E51(out)
        out = F.leaky_relu(out)
        out = self.conv3E52(out)
        out = self.dropoutE5(out)
        out = self.inorm3E52(out)
        out = F.leaky_relu(out)
        out = self.conv3E53(out)
        out = torch.add(out, sum5)
        out = self.inorm3E53(out)
        out = F.leaky_relu(out)

        # Level 0 local layer
        out = self.upscale(out)
        out = self.convD01(out)
        out = self.inorm3D01(out)
        out = F.leaky_relu(out)

        # Level 1 local layer
        out = torch.cat([out, skip4], dim=1)
        out = F.leaky_relu(self.inorm3D11(self.convD11(out)))
        out = F.leaky_relu(self.inorm3D12(self.convD12(out)))
        out = self.upscale(out)
        out = F.leaky_relu(self.inorm3D13(self.convD13(out)))

        # Level 2 local layer
        out = torch.cat([out, skip3], dim=1)
        out = F.leaky_relu(self.inorm3D21(self.convD21(out)))
        segment2 = self.seg2(out)
        segment2 = self.upscale(segment2)
        out = F.leaky_relu(self.inorm3D22(self.convD22(out)))
        out = self.upscale(out)
        out = F.leaky_relu(self.inorm3D23(self.convD23(out)))

        # Level 3 local layer
        out = torch.cat([out, skip2], dim=1)
        out = F.leaky_relu(self.inorm3D31(self.convD31(out)))
        segment3 = self.seg3(out)
        out = F.leaky_relu(self.inorm3D32(self.convD32(out)))
        out = self.upscale(out)
        out = F.leaky_relu(self.inorm3D33(self.convD33(out)))

        # Level 4 local layer
        out = torch.cat([out, skip1], dim=1)
        out = F.leaky_relu(self.inorm3D4(self.conv3D41(out)))

        # Segmentation layers
        segementResult = torch.add(segment2, segment3)
        segementResult = self.upscale(segementResult)

        # Output layers
        out = self.convOut(out)
        out = torch.add(out, segementResult)
        out = self.sigmoid(out)
        
        return out
