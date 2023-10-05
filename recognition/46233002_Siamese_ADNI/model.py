import torch
import torch.nn as nn
import torchvision.transforms.functional as F

class Siamese(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super(Siamese, self).__init__()

    def forward_one(self, x):

    def forward(self, x1, x2):