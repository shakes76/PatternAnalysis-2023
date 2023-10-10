import numpy as np
import torch
from torch_geometric.data import Data

features = torch.Tensor(np.load('facebook_large/features.npy'))
edges = np.rot90(np.load('facebook_large/edges.npy'), 1)
edges = torch.Tensor(edges.copy())
targets = torch.Tensor(np.load('facebook_large/target.npy'))

test = torch.Tensor(np.array(([1,2,3], [3,4,5])))

data = Data(x=features, edge_index=edges, edge_attr=None, y=targets)