import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    '''
    Constructs GCN
    '''
    def __init__(self, hidden_channels):
        super().__init__()
        self.dataset_num_features = 128
        self.dataset_num_classes = 4
        
        torch.manual_seed(42)
        self.conv1 = GCNConv(self.dataset_num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, self.dataset_num_classes)
        
    def forward(self, x, edge_index):
        print('inside forward')
        x = self.conv1(x, edge_index)
        print('finished conv1')
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        print('finished conv2')
        return x

# model = GCN(hidden_channels=16)
# print(model)