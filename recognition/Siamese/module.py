#This is for the setting up the components of the Siamese model

import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision import datasets
import torch.utils as util_data
import torch
import torch.nn as nn

#device configuration
device = torch.device("cuda")

def cfg(layer):
    if layer == 'VGG2' :
        return [32, 32, 'M']

#define Siamese model
class Siamese(nn.Module):
    def __init__(self, layers, in_channels = 1, classes = 2):
        super().__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.features = self.make_layers(cfg[layers], self.in_channels)
        self.classifier = nn.Linear(512, self.nbr_classes)
    
    def make_layers(self, cfg, in_channels):
        layers = []
        for x in cfg:
            if x == 'M':
                layers +=  [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [torch.nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           torch.nn.BatchNorm2d(x),
                           torch.nn.ReLU(inplace=True)]
                in_channels = x
        layers += [torch.nn.AvgPool2d(kernel_size=1, stride=1)]
        return torch.nn.Sequential(*layers)
    
    #Forward takes two input images to compare similarity
    def forward(self, x1, x2):
        output1 = self.features(x1)
        output2 = self.features(x2)
        output1 = output1.view(output1.size(0), -1)
        output1 = self.classifier(output1)
        output2 = output2.view(output2.size(0), -1)
        output2 = self.classifier(output2)
        return output1, output2