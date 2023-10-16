import torch
import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, in_channels=1, out_channels=256):
        super(Encoder, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=10, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Linear(173056, out_channels)
    
    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size()[0], -1)
        out = self.fc(out).clone()
        return out

class Classifier(nn.Module):
    def __init__(self, in_channels=256, out_channels=2):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(in_channels, 256)
        self.fc2 = nn.Linear(256, out_channels)

    def forward(self, x):
        out = torch.relu(self.fc1(x))
        out = torch.sigmoid(self.fc2(out))
        return out