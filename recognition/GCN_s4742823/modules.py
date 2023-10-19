'''
This file contains the Model class which is used to solve this task, as well as
the GNNLayer module which performs the graph convolution in it.
'''
import torch

# -------------
# This class was taken from Gayan Kulatilleke's lecture on GNNs during the Symposium,
# also available here: https://github.com/gayanku/SCGC/blob/main/models.py
# -------------
class GNNLayer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.parameter.Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=True):
        support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        if active:
            output = torch.nn.functional.relu(output)
        return output

class Model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, dropout_prob):
        super(Model, self).__init__()
        self.conv1 = GNNLayer(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = GNNLayer(hidden_channels, num_classes)
        self.bn2 = torch.nn.BatchNorm1d(num_classes)
        self.dropout = torch.nn.Dropout(p=dropout_prob)

    def forward(self, data):
        x, adj = data.x, data.adj
        x = self.conv1(x, adj)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, adj)
        x = self.bn2(x)
        return x