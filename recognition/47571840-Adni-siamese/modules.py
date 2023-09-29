import torch.nn as nn
import os
import numpy as np
import random
from torchvision import datasets, transforms
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from PIL import Image

class SubNetwork(nn.Module):
    def __init__(self, height, width):
        super(SubNetwork, self).__init__()

        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(height * width, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.dense_layers(x)
        return x


def distance_layer(out1, out2):
    return torch.sqrt(torch.sum((out1 - out2) ** 2, dim=1, keepdim=True))

def contrastive_loss(y_pred, y_true, margin=1.0):
    # y_pred is the output of the SiameseNetwork, between 0 and 1
    # y_true is 0 if images are similar, 1 if they are different
    square_pred = y_pred ** 2
    margin_square = torch.clamp(margin - y_pred, min=0) ** 2
    loss = (1 - y_true) * square_pred + y_true * margin_square
    return torch.mean(loss)



class SiameseNetwork(nn.Module):
    def __init__(self, height, width):
        super(SiameseNetwork, self).__init__()

        self.subnetwork = SubNetwork(height, width)
        self.batchnorm = nn.BatchNorm1d(1)
        self.fc = nn.Linear(1, 1)  # The output is a single scalar (distance)

    def forward(self, img1, img2):
        out1 = self.subnetwork(img1)
        out2 = self.subnetwork(img2)
        distance = distance_layer(out1, out2)
        distance = self.batchnorm(distance)
        out = torch.sigmoid(self.fc(distance))
        return out
