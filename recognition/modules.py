import torch
import torch.nn as nn

# Module structure
class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

    def forward(self, x):
        return x
