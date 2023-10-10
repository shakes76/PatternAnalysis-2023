import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit


def load_data():
    features = torch.Tensor(np.load('facebook_large/features.npy'))
    edges = np.rot90(np.load('facebook_large/edges.npy'), 1)
    edges = torch.Tensor(edges.copy())
    targets = torch.Tensor(np.load('facebook_large/target.npy'))
    
    data = Data(x=features, edge_index=edges, edge_attr=None, y=targets)
    
    return data

def test_train():
    features = torch.Tensor(np.load('facebook_large/features.npy'))
    edges = np.rot90(np.load('facebook_large/edges.npy'), 1)
    edges = torch.Tensor(edges.copy())
    targets = torch.Tensor(np.load('facebook_large/target.npy'))
    
    data = Data(x=features, edge_index=edges, edge_attr=None, y=targets)
    transform = RandomLinkSplit(is_undirected=True)
    return transform(data)

data = load_data()
train_data, val_data, test_data = test_train()
print(val_data)