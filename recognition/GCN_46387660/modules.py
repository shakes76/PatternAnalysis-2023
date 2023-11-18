'''
Create a GCN model
'''
import torch
import torch.nn as nn
from torch_geometric.nn.conv import GCNConv

class GCN(nn.Module):
    def __init__(self, in_channels=128, num_classes=4):
        super(GCN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        # Make two hidden layers and one output layer
        self.layer1 = GCNConv(in_channels, 256)
        self.layer2 = GCNConv(256, 256)
        self.out = nn.Linear(256, num_classes)
    
    def forward(self, x, edge_index):
        # Use relu as the activation function
        z = self.layer1(x, edge_index)
        z = z.relu()
        z = self.layer2(z, edge_index)
        z = z.relu()
        z = self.out(z)
        return z