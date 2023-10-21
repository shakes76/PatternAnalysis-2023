import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
    


class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.feature_num = 64
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        self.first_down = DoubleConv(in_channels, self.feature_num)
        self.second_down = DoubleConv(self.feature_num, self.feature_num*2)
        self.third_down = DoubleConv(self.feature_num*2, self.feature_num*4)
        self.fourth_down = DoubleConv(self.feature_num*4, self.feature_num*8)
        self.fifth_down = DoubleConv(features[-1], features[-1]*2)
        #DoubleConv(self.feature_num*8, self.feature_num*16)
        #LAYER 1
        #LAYER 2
        #LAYER 3
        #LAYER 4
        #LAYER 5
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        #LAYER 1
        x = self.first_down(x)
        skip_connections.append(x)
        x = self.pool(x)
        #LAYER 2
        x = self.second_down(x)
        skip_connections.append(x)
        x = self.pool(x)
        #LAYER 3
        x = self.third_down(x)
        skip_connections.append(x)
        x = self.pool(x)
        #LAYER 4
        x = self.fourth_down(x)
        skip_connections.append(x)
        x = self.pool(x)
        #LAYER 5
        x = self.fifth_down(x)
        '''
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        '''
        #x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

def test():
    x = torch.randn((3, 1, 161, 161))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()