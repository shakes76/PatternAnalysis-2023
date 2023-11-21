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
        self.conv1 = GCNConv(self.dataset_num_features, 100)
        self.conv2 = GCNConv(100, 64)
        self.conv3 = GCNConv(64, 32)
        self.final_conv = GCNConv(32, self.dataset_num_classes)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.final_conv(x, edge_index)
        return x

# model = GCN(hidden_channels=16)
# print(model)