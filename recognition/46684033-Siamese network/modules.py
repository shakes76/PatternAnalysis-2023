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
        self.linear = nn.Linear(50, 1)
        self.batchNorm1 = nn.BatchNorm2d(50)
        self.dropout = nn.Dropout(p=0.5)
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
            #nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            #nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            # nn.ReLU(),
            # nn.Linear(128, 64)
        )

    def forward_once(self, x):
        x = self.sequence(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x

    def forward(self, x, y, z=None):
        #     #BCELoss
        #     x = self.forward_once(x)
        #     y = self.forward_once(y)
        #     distance = torch.abs(y - x).view(64,50,1,1)
        #     output = self.batchNorm1(distance)
        #     output = torch.flatten(output, start_dim=1)
        #     output = self.linear(output)
        #     output = self.sigmoid(output)
        #     return output

        # contrastive Loss
        x = self.forward_once(x)
        y = self.forward_once(y)
        return x, y

        # #triplet loss
        # x = self.forward_once(x)
        # y = self.forward_once(y)
        # z = self.forward_once(z)
        # return x,y,z


class Classifier(nn.Module):
    def __init__(self, siamese):
        super(Classifier, self).__init__()
        self.siamese = siamese
        self.sigmoid = nn.Sigmoid()
        self.linear1 = nn.Linear(128, 64)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64,2)
        self.dropout = nn.Dropout(p=0.5)


    def forward(self, x):
        x = self.siamese.forward_once(x)
        x = self.relu(x)
        x = self.linear1(x)
        #x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

#TODO:
# try Siamese network output 128 features
# classifier try add two linear layer, add batchnorm and dropout layer within
# try the same data augmentation ramdoncrop on classifier and siamese
# try triplet loss