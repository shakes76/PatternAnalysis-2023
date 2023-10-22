import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class UNet(nn.Module):
    """
    Improved unet from the paper https://arxiv.org/pdf/1802.10508v1.pdf 
    Modified from https://github.com/mcost45/ISICs-improved-unet/blob/main/layers_model.py
    """
    def __init__(self, in_channels=3, out_channels=1, base_features=16):
        super(UNet, self).__init__()

        self.sigmoid = nn.Sigmoid()

        self.x1 = nn.Conv2d(in_channels, base_features, 3, 1, 1)
        self.x2 = nn.InstanceNorm2d(base_features)
        self.x3 = nn.LeakyReLU()
        self.x4 = self.ContexModule(base_features, base_features)

        self.x6 = nn.Conv2d(base_features, base_features*2, 3, stride=2, padding=1)
        self.x7 = nn.InstanceNorm2d(base_features*2)
        self.x8 = nn.LeakyReLU()
        self.x9 = self.ContexModule(base_features*2, base_features*2)

        self.x11 = nn.Conv2d(base_features*2, base_features*4, 3, stride=2, padding=1)
        self.x12 = nn.InstanceNorm2d(base_features*4)
        self.x13 = nn.LeakyReLU()
        self.x14 = self.ContexModule(base_features*4, base_features*4)

        self.x16 = nn.Conv2d(base_features*4, base_features*8, 3, stride=2, padding=1)
        self.x17 = nn.InstanceNorm2d(base_features*8)
        self.x18 = nn.LeakyReLU()
        self.x19 = self.ContexModule(base_features*8, base_features*8)

        self.x21 = nn.Conv2d(base_features*8, base_features*16, 3, stride=2, padding=1)
        self.x22 = nn.InstanceNorm2d(base_features*16)
        self.x23 = nn.LeakyReLU()
        self.x24 = self.ContexModule(base_features*16, base_features*16)
        self.x26 = self.UpsampleModule(base_features*16, base_features*8)

        self.x28 = self.LocalizationModule(base_features*16, base_features*8)
        self.x29 = self.UpsampleModule(base_features*8, base_features*4)

        self.x31 = self.LocalizationModule(base_features*8, base_features*4)
        self.x32 = self.UpsampleModule(base_features*4, base_features*2)     

        self.x34 = self.LocalizationModule(base_features*4, base_features*2)
        self.x35 = self.UpsampleModule(base_features*2, base_features) 

        self.x37 = nn.Conv2d(base_features*2, base_features*2, kernel_size=1)    
        self.x38 = nn.InstanceNorm2d(base_features*2)
        self.x39 = nn.LeakyReLU()

        self.u1 = self.UpsampleModule(base_features*4, base_features*2)

        self.u2 = self.UpsampleModule(base_features*2, base_features*2)

        self.output = nn.Conv2d(base_features*2, out_channels, 1)

    def ContexModule(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(),
        )
    
    def UpsampleModule(self, in_channels, out_channels):
    	return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
			nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
			nn.InstanceNorm2d(out_channels),
			nn.LeakyReLU(),
        )
    
    def LocalizationModule(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(),
        )
    

    def forward(self, x):

        out = self.x1(x)
        residual = out
        out = self.x2(out)
        out = self.x3(out)
        out = self.x4(out)
        out += residual
        hold_over_0 = out

        out = self.x6(out)
        out = self.x7(out)
        out = self.x8(out)
        residual = out
        out = self.x9(out)
        out += residual
        hold_over_1 = out

        out = self.x11(out)
        out = self.x12(out)
        out = self.x13(out)
        residual = out
        out = self.x14(out)
        out += residual
        hold_over_2 = out

        out = self.x16(out)
        out = self.x17(out)
        out = self.x18(out)
        residual = out
        out = self.x19(out)
        out += residual
        hold_over_3 = out

        out = self.x21(out)
        out = self.x22(out)
        out = self.x23(out)
        residual = out
        out = self.x24(out)
        out += residual  
        out = self.x26(out)     

        out = torch.cat([hold_over_3, out], dim=1)
        out = self.x28(out)
        out = self.x29(out)

        out = torch.cat([hold_over_2, out], dim=1)
        out = self.x31(out)
        segment1 = out
        out = self.x32(out)

        out = torch.cat([hold_over_1, out], dim=1)
        out = self.x34(out)
        segment2 = out
        out = self.x35(out)

        out = torch.cat([hold_over_0, out], dim=1)
        out = self.x37(out)
        out = self.x38(out) 
        out = self.x39(out)

        seg1 = self.u1(segment1)
        seg2 = segment2
        seg2 += seg1
        seg2 = self.u2(seg2)
        seg3 = out
        seg3 += seg2

        output = self.output(seg3)
        return self.sigmoid(output)

    
        
    
def test():
    x =  torch.randn((1, 3, 96, 128))
    model = UNet(in_channels=3, out_channels=1)
    preds = model(x)

    print(x.shape)
    print(preds.shape)


if __name__ == "__main__":
    test()