#This is for the setting up the components of the Siamese model

import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision import datasets
import torch.utils as util_data
import torch
import torch.nn as nn

def cfg(layer):
    if layer == 'VGG2' :
        return [32, 32, 'M']
    if layer == 'VGG4' :
        return [32, 32, 'M', 64, 64]
    if layer == 'VGG13' :
        return [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    if layer == 'VGG16' :
        return [64, 64, 'M', 128, 128 , 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    if layer == 'VGG19' :
        return [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]

#define Siamese model
class Siamese(nn.Module):
    def __init__(self, layers, in_channels, classes):
        super().__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.features = self.make_layers(cfg(layers), self.in_channels)
        self.classifier = nn.Linear(8192, self.classes)
    
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
        output1 = output1.view(output1.size(0), -1)
        output1 = self.classifier(output1)
        output2 = self.features(x2)
        output2 = output2.view(output2.size(0), -1)
        output2 = self.classifier(output2)
        return output1, output2

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, img1, img2, label):
        distance = torch.nn.functional.pairwise_distance(img1, img2, keepdim=True) #distance finds similarities between images
        reverse_dist = self.margin - distance #reverse distance find's difference between images
        reverse_dist = torch.clamp(reverse_dist, min=0.0)
        #if label is 1, returns distance, else if label is 0 returns reverse distance
        return torch.mean(torch.pow(reverse_dist, 2) * (abs(label-1)) + ((torch.pow(distance, 2)) * label))