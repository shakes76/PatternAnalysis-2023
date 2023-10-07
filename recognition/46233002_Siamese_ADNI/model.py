import torch
import torch.nn as nn
import torchvision.transforms.functional as F

class Siamese(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super(Siamese, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=10, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.PRelu(),
            nn.Linear(256, 256),
            nn.PRelu(),
            nn.Linear(256, out_channels),
        )

    def forward_one(self, x):
        out = self.conv(x)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out

    def forward(self, x1, x2, x3):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        out3 = self.forward_one(x3)
        return out1, out2, out3      

class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, pos, neg):
        dist_pos = (anchor - pos).pow(2).sum(1)
        dist_neg = (anchor - neg).pow(2).sum(1)
        loss = F.relu(dist_pos - dist_neg + self.margin)
        return loss.sum() # loss.mean()