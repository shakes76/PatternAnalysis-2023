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
        self.sequence = nn.Sequential(
            nn.Conv2d(1, 32, (5, 5), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(4),

            nn.Conv2d(32, 64, (3, 3), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3),

            nn.Conv2d(64, 128, (3, 3), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

        )
        self.fc = nn.Sequential(
            nn.Linear(1152, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

    def forward_once(self, x):
        x = self.sequence(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x

    def forward(self, x, y):
        # contrastive Loss
        x = self.forward_once(x)
        y = self.forward_once(y)
        return x, y


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64,2)

    def forward(self, y):
        x = self.linear1(y)
        x = self.relu(x)
        x = self.linear2(x)
        return x