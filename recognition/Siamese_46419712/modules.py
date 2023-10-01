import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

class SiameseModel(nn.Module):

    def __init__(self):
        super(SiameseModel, self).__init__()

    def forward(self, x):
        output = x
        return output

