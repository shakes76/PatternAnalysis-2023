import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    # input for command is the number of features in and the number of classes out
    def __init__(self, in_channels=128, num_classes=4):
        super(GCN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.layer1 = nn.GCNConv(in_channels, 256)
        self.layer2 = nn.GCNConv(256, 256)
        self.out = nn.Linear(256, num_classes)
        # use xavier uniform as it is between with deep neural networks with many layers and many nodes
        # Should make too much of a difference but might help something
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.zeros_(self.layer1.bias)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.zeros_(self.layer2.bias)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)
    
    def forward(self, x, edge_index):
        z = self.layer1(x, edge_index)
        z = z.relu()
        # Add dropout if overfitting is an issue
        #z = nn.functional.dropout(z, p=0.5, training=self.training)
        z = self.layer2(z, edge_index)
        z = z.relu()
        #z = nn.functional.dropout(z, p=0.5, training=self.training)
        z = self.out(z)
        return z
        

        

