import torch
from torch_geometric.nn import GCNConv

class Model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, dropout_prob):
        super(Model, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        self.bn2 = torch.nn.BatchNorm1d(num_classes)
        self.dropout = torch.nn.Dropout(p=dropout_prob)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        return x