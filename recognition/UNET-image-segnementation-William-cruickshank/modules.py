import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F

class FirstStep(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FirstStep, self).__init__()
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        self.ContextModule = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
        )

    def forward(self, x):
        intermediate = self.initial_conv(x)
        out = self.ContextModule(intermediate)
        
        summed_output = intermediate + out
        return summed_output

class MainSteps(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MainSteps, self).__init__()
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        self.ContextModule = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
        )

    def forward(self, x):
        intermediate = self.initial_conv(x)
        out = self.ContextModule(intermediate)
        
        summed_output = intermediate + out
        return summed_output

class UpSampleStep(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampleStep, self).__init__()
        
        self.LastCon = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        
        out = F.interpolate(x, scale_factor=2, mode='nearest')
        out = self.LastCon(out)
        
        return out

class LocalizationStep(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LocalizationStep, self).__init__()
        
        self.Localization = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.Localization(x)
        return out

class SegmentationLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegmentationLayer, self).__init__()
        
        self.Localization = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.Localization(x)
        return out

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[16, 32, 64, 128],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        #self.pool = ContextConnector()
        #self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        self.downs.append(FirstStep(in_channels, features[0]))
        in_channels = features[0]
        for feature in features[1:]:
            self.downs.append(MainSteps(in_channels, feature))
            in_channels = feature

        self.bottleneck = MainSteps(features[-1], features[-1]*2)
        
        # Up part of UNET
        self.ups.append(UpSampleStep(features[3]*2, features[3]))
        self.ups.append(LocalizationStep(features[3]*2, features[3])) # 128
        self.ups.append(UpSampleStep(features[2]*2, features[2]))
        self.ups.append(LocalizationStep(features[2]*2, features[2])) # 64 # id 3
        self.ups.append(UpSampleStep(features[1]*2, features[1]))
        self.ups.append(LocalizationStep(features[1]*2, features[1])) # 32 # id 5
        self.ups.append(UpSampleStep(features[0]*2, features[0]))
        
        self.prefinal_conv = nn.Sequential(
            nn.Conv2d(features[0], features[1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(features[1]),
            nn.ReLU(inplace=True),
        )
        self.final_conv3 = nn.Conv2d(features[1], out_channels, kernel_size=1)
        self.final_conv2 = nn.Conv2d(features[1], out_channels, kernel_size=1)
        self.final_conv1 = nn.Conv2d(features[2], out_channels, kernel_size=1)
        self.final_conv = nn.Conv2d(features[1], out_channels, kernel_size=1)
        self.final_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            #x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        

        for idx in range(0, len(self.ups) + 2, 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            if (idx == 6):
                break
            x = self.ups[idx+1](concat_skip)
            if (idx == 2):
                final_conv_out1 = self.final_conv1(x)
            elif (idx == 4):
                final_conv_out1 = F.interpolate(final_conv_out1, scale_factor=2, mode='nearest')
                final_conv_out2 = self.final_conv2(x)
                if final_conv_out1.shape != final_conv_out2.shape:
                    final_conv_out1 = TF.resize(x, final_conv_out2.shape[2:])
                final_conv_out2 = final_conv_out2 + final_conv_out1
        
        x = self.prefinal_conv(x)
        final_conv_out2 = F.interpolate(final_conv_out2, scale_factor=2, mode='nearest')
        final_conv_out3 = self.final_conv3(x)
        if final_conv_out2.shape != final_conv_out3.shape:
            final_conv_out2 = TF.resize(x, final_conv_out3.shape[2:])
        final_conv_out3 = final_conv_out3 + final_conv_out2
        return self.final_conv(final_conv_out3)