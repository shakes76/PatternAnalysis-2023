import dataset
import torch
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(dataset.number_features, 2 * hidden_channels)
        self.conv2 = GCNConv(2 * hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, dataset.number_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        return x
