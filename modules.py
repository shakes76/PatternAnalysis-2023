import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv



class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        dataset_num_features = 128
        dataset_num_classes = 4
        
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(dataset_num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset_num_classes)

model = GCN(hidden_channels=16)
print(model)