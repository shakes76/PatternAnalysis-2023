import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GCNConv(nn.Module):
    def __init__(self, sample_size, in_channels, out_channels):
        super().__init__()
        self.sample_size = sample_size
        self.weight = Parameter(torch.FloatTensor(in_channels, out_channels))
        self.bias = Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        adj = adj.coalesce()
        out_degree = adj.sum(dim=1).to_dense().sqrt_()

        # Ensuring out_degree is in the correct shape
        out_degree_inv = 1 / out_degree.view(-1)

        # Generating a sparse diagonal matrix
        indices = torch.arange(len(out_degree)).unsqueeze(0).repeat(2, 1).to(x.device)
        norm_diag_matrix = torch.sparse_coo_tensor(indices, out_degree_inv, (x.size(0), x.size(0)), device=x.device)

        # Properly compute the normalized adjacency matrix and feature support
        support = torch.sparse.mm(adj, x)
        support = torch.sparse.mm(norm_diag_matrix, support)
        output = torch.mm(support, self.weight) + self.bias

        return output


class multi_GCN(nn.Module):
    def __init__(self, sample_size, features_size, classes_size, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(sample_size, features_size, 2 * hidden_channels)
        self.conv2 = GCNConv(sample_size, 2 * hidden_channels, hidden_channels)
        self.conv3 = GCNConv(sample_size, hidden_channels, hidden_channels)
        self.conv4 = GCNConv(sample_size, hidden_channels, classes_size)

        # Adding Batch Normalization layers
        self.bn1 = nn.BatchNorm1d(2 * hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(self.bn1(x))
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(self.bn2(x))
        x = F.dropout(x, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.relu(self.bn3(x))
        x = F.dropout(x, training=self.training)

        x = self.conv4(x, edge_index)
        return F.log_softmax(x, dim=1)
