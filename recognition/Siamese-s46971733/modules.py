# Imports
import time
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2
import os

class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        # Stride is the 'jump' from one element to the next.
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Resnet's shortcut makes it possible to skip layers which allows networks with much deeper layers.
        # Fixes problem of vanishing gradient -- where gradient gets smaller as loss function tries to minimise
        #                                              If it disappears, cannot be optimised anymore.

        # Resnet 'carries' features to the output, ensuring they are not loss.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,
                          self.expansion*out_channels,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(self.expansion*out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3,
                               64,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        #self.linear = nn.Linear(512*block.expansion, num_classes)
        #self.linear = nn.Linear(512 * block.expansion, 10)
        self.linear = nn.Linear(512 * block.expansion, 64)

        # self.triplet_loss = nn.Sequential(
        #     nn.Linear(num_classes, 2))
        # self.triplet_loss = nn.Sequential(
        #                 nn.Linear(10, num_classes),
        #                 nn.ReLU())

    def _make_layer(self, block, out_channels, num_blocks, stride):
        # Basically makes one block downsample and the other compute convolutional layer.
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # Take Average of 4 by 4 image.
        out = F.avg_pool2d(out, 4)
        # Flattens layer.
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        #out = self.triplet_loss(out)
        #out = self.triplet_loss(out)

        return out

class classifer(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Linear(16, 2)
        )

        self.activation = nn.Sigmoid()

    def forward(self, x):
        out = self.layer(x)
        out = self.activation(out)

        return out


def Resnet():
    return ResNet(Block, [2, 2, 2, 2])

def Resnet34():
    return ResNet(Block, [3, 4, 6, 3])


