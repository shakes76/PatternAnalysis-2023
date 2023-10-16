"""
Author: William Barker
SN: 4642798
This script defines a Graph Convolutional Network (GCN) along with a custom 
GCN convolution layer, implementing the propagation rule as described in the 
Graph Convolutional Networks (GCN) paper (Kipf and Welling, 2017).
"""
import torch
import torch.nn.functional as F
from torch.nn import Parameter


class GCNConv(torch.nn.Module):
    """
    A class which represents a single convolution layer of the GCN
    """

    def __init__(self, sample_size, in_channels, out_channels):
        super().__init__()
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = Parameter(torch.empty(out_channels))
        self.weight = Parameter(torch.FloatTensor(in_channels, out_channels))
        self.reset_parameters()
        torch.nn.init.xavier_uniform_(self.weight)

    def reset_parameters(self):
        self.bias.data.zero_()

    def forward(self, x, edges_sparse):
        # calulate inverse square root of the degree matrix
        out_degree = edges_sparse.sum(dim=1).to_dense()
        deg_inv_sqrt = out_degree.pow(-0.5)

        # convert to sparse matrix
        diag_indicies = [[i, i] for i in range(out_degree.size(0))]
        diag_indicies = torch.tensor(diag_indicies)
        D = torch.sparse_coo_tensor(
            diag_indicies.t(),
            deg_inv_sqrt,
            torch.Size([self.sample_size, self.sample_size]),
        )

        # Compute D^(-1/2) A D^(-1/2) X
        val1 = torch.sparse.mm(D, edges_sparse)
        val2 = torch.sparse.mm(val1, D)
        val3 = torch.sparse.mm(val2, x)
        out = torch.mm(val3, self.weight)

        # Apply bias
        out += self.bias

        return out


class GCN(torch.nn.Module):
    """
    Class representing the 3 layer GCN network trained against the Facebook dataset
    """

    def __init__(self, sample_size, number_features, number_classes, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(sample_size, number_features, 2 * hidden_channels)
        self.conv2 = GCNConv(sample_size, 2 * hidden_channels, hidden_channels)
        self.conv3 = GCNConv(sample_size, hidden_channels, number_classes)

    def forward(self, x, edge_index):
        # First convolutional layer
        x = self.conv1(x, edge_index)
        # Activation Function
        x = x.relu()
        # Dropout to prevent overfitting
        x = F.dropout(x, p=0.5, training=self.training)
        # Second convolutional layer
        x = self.conv2(x, edge_index)
        # Activation Function
        x = x.relu()
        # Dropout to prevent overfitting
        x = F.dropout(x, p=0.5, training=self.training)
        # Third convolutional layer
        x = self.conv3(x, edge_index)
        # Softmax for classification of output
        return F.softmax(x, dim=1)
