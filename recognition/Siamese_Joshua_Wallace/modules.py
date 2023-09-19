import torch
import torch.nn as nn
import torch.nn.functional as F


class Siamese(nn.Module):

    def __init__(self):
        super(Siamese, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 10),  # 64@96*96
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64@48*48
            nn.Conv2d(64, 128, 7),
            nn.ReLU(inplace=True),  # 128@42*42
            nn.MaxPool2d(2),  # 128@21*21
            nn.Conv2d(128, 128, 4),
            nn.ReLU(inplace=True),  # 128@18*18
            nn.MaxPool2d(2),  # 128@9*9
            nn.Conv2d(128, 256, 5),
            nn.ReLU(inplace=True),  # 256@6*6
            nn.Conv2d(256, 256, 4),
            nn.ReLU(inplace=True)   # 256@24x24
        )
        self.liner = nn.Sequential(nn.Linear(400, 4096), nn.Sigmoid())
        self.out = nn.Linear(4096, 1)

    def forward(self, x1, x2):
        h1 = self.sub_forward(x1)
        h2 = self.sub_forward(x2)

        diff = torch.abs(h1 - h2)

        scores = self.out(diff)
        return scores

    def sub_forward(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.liner(x)
        return x