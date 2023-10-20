import torch
import torch.nn as nn

class Siamese(nn.Module):

    def __init__(self, in_channels=1, out_channels=128):
        super(Siamese, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=10, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Linear(80000, out_channels)
    
    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out

class Classifier(nn.Module):
    def __init__(self, in_channels=128, out_channels=2):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, out_channels),
        )

    def forward(self, x):
        out = torch.sigmoid(self.fc(x))
        return out