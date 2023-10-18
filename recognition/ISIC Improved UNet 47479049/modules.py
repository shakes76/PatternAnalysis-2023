import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, base_features=8):
        super(UNet, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.lrelu = nn.LeakyReLU()


        # Down 0
        self.initial = nn.Conv2d(in_channels, base_features, kernel_size=3, padding=1, bias=False)
        self.cm0 = self.ContexModule(base_features, base_features)

        # Down 1
        self.conv1 = nn.Conv2d(base_features, base_features*2, kernel_size=3, stride=2, padding=1, bias=False)
        self.cm1 = self.ContexModule(base_features*2, base_features*2)

        # Down 2
        self.conv2 = nn.Conv2d(base_features*2, base_features*4, kernel_size=3, stride=2, padding=1, bias=False)
        self.cm2 = self.ContexModule(base_features*4, base_features*4)

        # Down 3
        self.conv3 = nn.Conv2d(base_features*4, base_features*8, kernel_size=3, stride=2, padding=1, bias=False)
        self.cm3 = self.ContexModule(base_features*8, base_features*8)

        # Down 4
        self.conv4 = nn.Conv2d(base_features*8, base_features*16, kernel_size=3, stride=2, padding=1, bias=False)
        self.cm4 = self.ContexModule(base_features*16, base_features*16)

        # Up 0
        self.um0 = nn.ConvTranspose2d(base_features*16, base_features*8, kernel_size=3, stride=2, padding=1, output_padding=1)

        # Up 1
        self.lm1 = self.LocalizationModule(base_features*16, base_features*8)
        self.um1 =  nn.ConvTranspose2d(base_features*8, base_features*4, kernel_size=3, stride=2, padding=1, output_padding=1)

        # Up 2
        self.lm2 = self.LocalizationModule(base_features*8, base_features*4)
        self.segl2 = nn.Conv2d(base_features*4, out_channels, kernel_size=3, padding="same")
        self.um2 =  nn.ConvTranspose2d(base_features*4, base_features*2, kernel_size=3, stride=2, padding=1, output_padding=1)

        # Up 3
        self.lm3 = self.LocalizationModule(base_features*4, base_features*2)
        self.segl3 = nn.Conv2d(base_features*2, out_channels, kernel_size=3, padding="same")
        self.um3 =  nn.ConvTranspose2d(base_features*2, base_features, kernel_size=3, stride=2, padding=1, output_padding=1)

        # Up 4
        self.final_cov = nn.Conv2d(base_features*2, out_channels, kernel_size=3, stride=1, padding=1)
        self.soft_max = nn.Softmax(dim=1)


    def ContexModule(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.3),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(),
        )
    
    
    def LocalizationModule(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(),
        )
    
    
    


    

    def forward(self, x):

        # Down 0
        out = self.initial(x)
        residual = out
        out = self.cm0(out)
        out += residual
        concat_0 = out

        # Down 1
        out = self.conv1(out)
        residual = out
        self.lrelu(out)
        out = self.cm1(out)
        out += residual
        concat_1 = out

        # Down 2
        out = self.conv2(out)
        residual = out
        self.lrelu(out)
        out = self.cm2(out)
        out += residual
        concat_2 = out

        # Down 3
        out = self.conv3(out)
        residual = out
        self.lrelu(out)
        out = self.cm3(out)
        out += residual
        concat_3 = out

        # Down 4
        out = self.conv4(out)
        residual = out
        self.lrelu(out)
        out = self.cm4(out)
        out += residual

        # Up 0
        out = self.um0(out)

        # Up 1
        out = torch.cat([out, concat_3], dim=1)
        out = self.lm1(out)
        out = self.um1(out)

        # Up 2
        out = torch.cat([out, concat_2], dim=1)
        out = self.lm2(out)
        seg_2 = self.segl2(out)
        seg_2 = self.upsample(seg_2)
        out = self.um2(out)

        # Up 3
        out = torch.cat([out, concat_1], dim=1)
        out = self.lm3(out)
        seg_3 = self.segl3(out)
        out = self.um3(out)

        # Up 4
        out = torch.cat([out, concat_0], dim=1)
        out = self.final_cov(out)
        self.lrelu(out)

        seg_2_add_seg_3 = seg_2+seg_3
        seg_2_add_seg_3 = self.upsample(seg_2_add_seg_3)

        out += seg_2_add_seg_3


        return self.soft_max(out)
    
def test():
    x =  torch.randn((3, 1, 96, 128))
    model = UNet(in_channels=1, out_channels=1)
    preds = model(x)

    print(x.shape)
    print(preds.shape)


if __name__ == "__main__":
    test()