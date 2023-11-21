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
        self.linear = nn.Linear(512 * block.expansion, 64)

        self.lin_layer = nn.Sequential(
                                        nn.AvgPool2d(4, stride=2),
                                        nn.Dropout(p=0.1),
                                        nn.ReLU(),
                                        nn.Flatten(),
                                        nn.Linear(512, 512),
                                        nn.ReLU()
                                       )

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

        #out = self.lin_layer(out)

        #out = self.triplet_loss(out)

        return out

##########################
#       3D  Resnet       #
##########################


class Block3D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        # Stride is the 'jump' from one element to the next.
        super(Block3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels,
                               out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        # Resnet's shortcut makes it possible to skip layers which allows networks with much deeper layers.
        # Fixes problem of vanishing gradient -- where gradient gets smaller as loss function tries to minimise
        #                                              If it disappears, cannot be optimised anymore.

        # Resnet 'carries' features to the output, ensuring they are not loss.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels,
                          self.expansion*out_channels,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm3d(self.expansion*out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

###########################################################################################################
# Code inspired by Zheng, B et al. https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ipr2.12618
###########################################################################################################


class ResNet3D(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet3D, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv3d(1,
                               64,
                               kernel_size=7,
                               stride=2,
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.max_pool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # 4 Layers with increasing out channels.
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.linear1 = nn.Linear(25088 * block.expansion, 64)
        self.linear5 = nn.Linear(25088*block.expansion, 1000)
        self.linear6 = nn.Linear(25088, 512)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        # Creating convolutional layers
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial Convolution and Normalisation
        out = F.relu(self.bn1(self.conv1(x)))
        # Max Pooling
        out = self.max_pool1(out)

        # Resnet Convolution layers (with shortcuts)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # Flattens layer.
        out = out.view(out.size(0), -1)

        # Linear Layer and Activation
        out = F.relu(self.linear5(out))

        return out


class classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2560, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(),

            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 2)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(4096, 2048),
            #nn.BatchNorm1d(2048),
            nn.ReLU(),

            nn.Linear(2048, 1024),
            #nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 512),
            #nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(512, 1)
        )

        self.layer3 = nn.Sequential(

            nn.Linear(1000, 512),
            # nn.BatchNorm1d(2048),
            nn.ReLU(),

            nn.Linear(512, 128),
            # nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(128, 32),
            # nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(32, 1)
        )

        self.activation = nn.Sigmoid()

    def forward(self, x):
        out = self.layer3(x)
        out = self.activation(out)

        return out


def Resnet():
    return ResNet(Block, [2, 2, 2, 2])


def Resnet34():
    return ResNet(Block, [3, 4, 6, 3])


def Resnet3D():
    return ResNet3D(Block3D, [2, 2, 2, 2])


def Resnet3D_34():
    return ResNet3D(Block3D, [3, 4, 6, 3])
