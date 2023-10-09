##################################   utils.py   ##################################
import torch
from torch.nn import Module

class Loss(Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, x1, x2, y):
        difference = x0 - x1
        distance_square = torch.sum(torch.pow(difference, 2), 1)
        distance = torch.sqrt(distance_square)
        margin_distance = 1 - distance
        distance = torch.clamp(margin_distance, min=0.0)
        loss = y * distance_square + (1 - y) * torch.pow(distance, 2)
        return torch.sum(loss) / 2.0 / x1.size()[0]