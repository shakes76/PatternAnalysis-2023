import torch
import numpy as np
from torch.nn import Linear
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn = GCNConv(128, 3)
        self.linear = Linear(3, 4)

    def forward(self, x, edges):
        h = self.gcn(x, edges).relu()
        z = self.linear(h)
        return h, z
    
    def accuracy(self, true, predict):
        return (true == predict).sum() / len(predict)