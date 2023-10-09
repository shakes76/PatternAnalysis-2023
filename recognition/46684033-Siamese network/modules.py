# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F

class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)
        self.linear = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, 1)
        self.conv1 = nn.Conv2d(2, 1, 1)
        # self.conv2 = nn.Conv2d(32,64,(7,7),bias=False)
        # self.conv3 = nn.Conv2d(64,128,(4,4),bias=False)
        # self.conv4 = nn.Conv2d(128,1,(2,2))
        # self.maxpool = nn.MaxPool2d(4)
        # self.batchNorm1 = nn.BatchNorm2d(32)
        # self.batchNorm2 = nn.BatchNorm2d(64)
        # self.batchNorm3 = nn.BatchNorm2d(128)
        self.sequence = nn.Sequential(
            nn.Conv2d(3, 32, (10, 10),bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(4),

            nn.Conv2d(32, 64, (7, 7),bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(4),

            nn.Conv2d(64, 128, (4, 4),bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(4),

        )
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward_once(self, x):
        x = self.sequence(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x

    def forward(self, x, y,z=None):
        #BCELoss
        # x = self.forward_once(x)
        # y = self.forward_once(y)
        # output = F.pairwise_distance(x, y, keepdim=True)
        # output = self.sigmoid(output)
        # return output

        # contrastive Loss
        x = self.forward_once(x)
        y = self.forward_once(y)
        return x,y

        # #triplet loss
        # x = self.forward_once(x)
        # y = self.forward_once(y)
        # z = self.forward_once(z)
        # return x,y,z