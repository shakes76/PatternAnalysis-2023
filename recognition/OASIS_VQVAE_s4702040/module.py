import torch
from torch import nn
from torch.nn import functional as F

#Model
class ImprovedUNet(nn.Module):
    def __init__(self, in_channels, n_classes, base_n_filter = 4):
        super(ImprovedUNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter

        self.dropout3d = nn.Dropout2d(0.3)
        self.softmax = nn.Softmax(dim=1)
        self.upscale = nn.Upsample(scale_factor=2, mode='nearest')

        # Level 1 context layer
        self.conv3E11 = nn.Conv2d(self.in_channels, base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3d2 = nn.Conv2d(base_n_filter, base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3d3 = nn.Conv2d(base_n_filter, base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
        self.inorm3d = nn.InstanceNorm2d(self.base_n_filter)

        # Level 2 context
        self.conv3E21 = nn.Conv2d(base_n_filter, base_n_filter*2, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv3E22 = nn.Conv2d(base_n_filter*2, base_n_filter*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3E23 = nn.Conv2d(base_n_filter*2, base_n_filter*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.inorm3E2 = nn.InstanceNorm2d(self.base_n_filter*2)

        # Level 3 context
        self.conv3E31 = nn.Conv2d(base_n_filter*2, base_n_filter*4, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv3E32 = nn.Conv2d(base_n_filter*4, base_n_filter*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3E33 = nn.Conv2d(base_n_filter*4, base_n_filter*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.inorm3E3 = nn.InstanceNorm2d(self.base_n_filter*4)

        # Level 4 context
        self.conv3E41 = nn.Conv2d(base_n_filter*4, base_n_filter*8, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv3E42 = nn.Conv2d(base_n_filter*8, base_n_filter*8, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3E43 = nn.Conv2d(base_n_filter*8, base_n_filter*8, kernel_size=3, stride=1, padding=1, bias=False)
        self.inorm3E4 = nn.InstanceNorm2d(self.base_n_filter*8)

        # Level 5 context
        self.conv3E51 = nn.Conv2d(base_n_filter*8, base_n_filter*16, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv3E52 = nn.Conv2d(base_n_filter*16, base_n_filter*16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3E53 = nn.Conv2d(base_n_filter*16, base_n_filter*16, kernel_size=3, stride=1, padding=1, bias=False)
        self.inorm3E5 = nn.InstanceNorm2d(self.base_n_filter*16)

        # Level 0 local
        self.convD01 = nn.Conv2d(base_n_filter*16, base_n_filter*8, kernel_size=3, stride=1, padding=1, bias=False)
        self.inorm3D01 = nn.InstanceNorm2d(self.base_n_filter*8)
        #self.convD02 = nn.Conv2d(base_n_filter*16, base_n_filter*8, kernel_size=1, stride=1, padding=0, bias=False)
        #self.inorm3D02 = nn.InstanceNorm2d(self.base_n_filter*8)

        # Level 1 local
        #local mod
        self.convD11 = nn.Conv2d(base_n_filter*16, base_n_filter*16, kernel_size=3, stride=1, padding=1, bias=False)
        self.inorm3D11 = nn.InstanceNorm2d(self.base_n_filter*16)
        self.convD12 = nn.Conv2d(base_n_filter*16, base_n_filter*8, kernel_size=1, stride=1, padding=0, bias=False)
        self.inorm3D12 = nn.InstanceNorm2d(self.base_n_filter*8)

        #upsample mod
        self.convD13 = nn.Conv2d(base_n_filter*8, base_n_filter*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.inorm3D13 = nn.InstanceNorm2d(self.base_n_filter*4)

        # Level 2 local
        #local mod
        self.convD21 = nn.Conv2d(base_n_filter*8, base_n_filter*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.inorm3D21 = nn.InstanceNorm2d(self.base_n_filter*4)
        self.convD22 = nn.Conv2d(base_n_filter*4, base_n_filter*4, kernel_size=1, stride=1, padding=0, bias=False)
        self.inorm3D22 = nn.InstanceNorm2d(self.base_n_filter*4)

        #upsample mod
        self.convD23 = nn.Conv2d(base_n_filter*4, base_n_filter*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.inorm3D23 = nn.InstanceNorm2d(self.base_n_filter*2)

        #segmentation layer
        self.seg2 = nn.Conv2d(base_n_filter*4, 1, kernel_size=3, stride=1, padding=1, bias=False)

        # Level 3 local
        #local mod
        self.convD31 = nn.Conv2d(base_n_filter*4, base_n_filter*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.inorm3D31 = nn.InstanceNorm2d(self.base_n_filter*2)
        self.convD32 = nn.Conv2d(base_n_filter*2, base_n_filter*2, kernel_size=1, stride=1, padding=0, bias=False)
        self.inorm3D32 = nn.InstanceNorm2d(self.base_n_filter*2)

        #upsample mod
        self.convD33 = nn.Conv2d(base_n_filter*2, base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
        self.inorm3D33 = nn.InstanceNorm2d(self.base_n_filter)

        #segmentation layer
        self.seg3 = nn.Conv2d(base_n_filter*2, 1, kernel_size=3, stride=1, padding=1, bias=False)

        # Level 4 localization layer
        self.conv3D41 = nn.Conv2d(base_n_filter*2, base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
        self.inorm3D4 = nn.InstanceNorm2d(self.base_n_filter)

        # Output layer
        self.convOut = nn.Conv2d(base_n_filter, 1, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        #  Level 1 context pathway
        out = self.conv3E11(x)
        sum1 = out
        out = F.leaky_relu(out)
        out = self.conv3d2(out)
        out = self.dropout3d(out)
        out = F.leaky_relu(out)
        out = self.conv3d3(out)
        out = torch.add(out, sum1)
        skip1 = F.leaky_relu(out)
        out = self.inorm3d(out)
        out = F.leaky_relu(out)

        # Level 2 context pathway
        #"""
        out = self.conv3E21(out)
        sum2 = out
        out = F.leaky_relu(out)
        out = self.conv3E22(out)
        out = self.dropout3d(out)
        out = F.leaky_relu(out)
        out = self.conv3E23(out)
        out = torch.add(out, sum2)
        skip2 = F.leaky_relu(out)
        out = self.inorm3E2(out)
        out = F.leaky_relu(out)
        #"""

        # Level 3 context pathway
        #"""
        out = self.conv3E31(out)
        sum3 = out
        out = F.leaky_relu(out)
        out = self.conv3E32(out)
        out = self.dropout3d(out)
        out = F.leaky_relu(out)
        out = self.conv3E33(out)
        out = torch.add(out, sum3)
        skip3 = F.leaky_relu(out)
        out = self.inorm3E3(out)
        out = F.leaky_relu(out)
        #"""

        # Level 4 context pathway
        #"""
        out = self.conv3E41(out)
        sum4 = out
        out = F.leaky_relu(out)
        out = self.conv3E42(out)
        out = self.dropout3d(out)
        out = F.leaky_relu(out)
        out = self.conv3E43(out)
        out = torch.add(out, sum4)
        skip4 = F.leaky_relu(out)
        out = self.inorm3E4(out)
        out = F.leaky_relu(out)
        #"""

        # Level 5 context pathway and level 0 local
        #"""
        out = self.conv3E51(out)
        sum5 = out
        out = F.leaky_relu(out)
        out = self.conv3E52(out)
        out = self.dropout3d(out)
        out = F.leaky_relu(out)
        out = self.conv3E53(out)
        out = torch.add(out, sum5)
        out = self.inorm3E5(out)
        out = F.leaky_relu(out)

        out = self.upscale(out)
        out = self.convD01(out)
        out = self.inorm3D01(out)
        out = F.leaky_relu(out)
        #out = self.convD02(out)
        #out = self.inorm3D02(out)
        #out = F.leaky_relu(out)
        #"""

        # Level 1 local
        out = torch.cat([out, skip4], dim=1)
        out = F.leaky_relu(self.inorm3D11(self.convD11(out)))
        out = F.leaky_relu(self.inorm3D12(self.convD12(out)))
        out = self.upscale(out)
        out = F.leaky_relu(self.inorm3D13(self.convD13(out)))

        # Level 2 local
        out = torch.cat([out, skip3], dim=1)
        out = F.leaky_relu(self.inorm3D21(self.convD21(out)))
        segment2 = self.seg2(out)
        segment2 = self.upscale(segment2)
        out = F.leaky_relu(self.inorm3D22(self.convD22(out)))
        out = self.upscale(out)
        out = F.leaky_relu(self.inorm3D23(self.convD23(out)))

        # Level 3 local
        out = torch.cat([out, skip2], dim=1)
        out = F.leaky_relu(self.inorm3D31(self.convD31(out)))
        segment3 = self.seg3(out)
        out = F.leaky_relu(self.inorm3D32(self.convD32(out)))
        out = self.upscale(out)
        out = F.leaky_relu(self.inorm3D33(self.convD33(out)))

        # Level 4 local
        out = torch.cat([out, skip1], dim=1)
        out = F.leaky_relu(self.inorm3D4(self.conv3D41(out)))

        segementResult = torch.add(segment2, segment3)
        segementResult = self.upscale(segementResult)

        # Output layers
        out = self.convOut(out)
        out = torch.add(out, segementResult)
        out = self.softmax(out)
        return out
