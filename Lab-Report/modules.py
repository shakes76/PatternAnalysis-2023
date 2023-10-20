import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels, n_class):
        super(UNet, self).__init__()
        self.n_class = n_class

        # Encoder
        self.e11 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.e12_bn = nn.BatchNorm2d(64)  # Batch normalization
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.e22_bn = nn.BatchNorm2d(128)  # Batch normalization
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.e32_bn = nn.BatchNorm2d(256)  # Batch normalization
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.e42_bn = nn.BatchNorm2d(512)  # Batch normalization
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.e52_bn = nn.BatchNorm2d(1024)  # Batch normalization
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.d12_bn = nn.BatchNorm2d(512)  # Batch normalization

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.d22_bn = nn.BatchNorm2d(256)  # Batch normalization

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.d32_bn = nn.BatchNorm2d(128)  # Batch normalization

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.d42_bn = nn.BatchNorm2d(64)  # Batch normalization

        # Output layer (Segmentation Layer)
        self.segmentation = nn.Conv2d(64, self.n_class, kernel_size=1)

        # Initialize weights with He initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        # Encoder
        xe11 = F.relu(self.e12_bn(self.e11(x)))
        xe12 = F.relu(self.e12_bn(self.e12(xe11)))
        xp1 = self.pool1(xe12)

        xe21 = F.relu(self.e22_bn(self.e21(xp1)))
        xe22 = F.relu(self.e22_bn(self.e22(xe21)))
        xp2 = self.pool2(xe22)

        xe31 = F.relu(self.e32_bn(self.e31(xp2)))
        xe32 = F.relu(self.e32_bn(self.e32(xe31)))
        xp3 = self.pool3(xe32)

        xe41 = F.relu(self.e42_bn(self.e41(xp3)))
        xe42 = F.relu(self.e42_bn(self.e42(xe41)))
        xp4 = self.pool4(xe42)

        xe51 = F.relu(self.e52_bn(self.e51(xp4)))
        xe52 = F.relu(self.e52_bn(self.e52(xe51)))

        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = F.relu(self.d12_bn(self.d11(xu11)))
        xd12 = F.relu(self.d12_bn(self.d12(xd11)))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = F.relu(self.d22_bn(self.d21(xu22)))
        xd22 = F.relu(self.d22_bn(self.d22(xd21)))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = F.relu(self.d32_bn(self.d31(xu33)))
        xd32 = F.relu(self.d32_bn(self.d32(xd31)))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = F.relu(self.d42_bn(self.d41(xu44)))
        xd42 = F.relu(self.d42_bn(self.d42(xd41)))

        # Output layer (Segmentation Layer)
        segmentation = torch.sigmoid(self.segmentation(xd42))

        return segmentation
