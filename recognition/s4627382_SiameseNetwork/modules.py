  # containing the source code of the components of your model.
# Each component must be implementated as a class or a function

import torch
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier
device = torch.device('cuda')

# Build CNN network and get its embedding vector
class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.conv = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # size: 256*240 -> 128*120

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # size: 128*120 -> 63*59
            )

        self.fc = nn.Sequential(
            nn.Linear(64*63*59, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 256),
            )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out


# construct the triplet loss
# formular: L = (1 - y) * 1/2 * D^2 + y * 1/2 * max(0, m - D)^2 
# where D = sample distance, m = margin, y = label, same: label = 0; diff, label = 1
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, img1, img2, label):
        # calculate euclidean distance  
        distance = (img1 - img2).pow(2).sum(1).sqrt()

        # calculate loss, use relu to ensure loss are non-negative
        loss_same = (1 - label) * 0.5 * (distance ** 2)
        loss_diff = label * 0.5 * torch.relu(self.margin - distance).pow(2)
        loss = loss_same + loss_diff

        return loss.mean()


# construct the siamese network
class SiameseNet(nn.Module):
    def __init__(self, embedding):
        super(SiameseNet, self).__init__()
        self.embedding = embedding

    def forward(self, img1, img2):
        emb1 = self.embedding(img1)
        emb2 = self.embedding(img2)
        return emb1, emb2

    def get_embedding(self, x):
        return self.embedding(x)

