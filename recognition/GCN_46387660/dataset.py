'''
Create the dataset from the given data
'''

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import random

# Load dataset from npz file
data = np.load('facebook.npz')

# allocate x, y and edge_index tensors
x = torch.from_numpy(data['features']).to(torch.float)
y = torch.from_numpy(data['target']).to(torch.long)
edge_index = torch.from_numpy(data['edges']).to(torch.long)
# changes the edge index to a list of index tuples
edge_index = edge_index.t().contiguous()

# Assign the fist 16000 (~70%) to the train set and the rest to the test set
train_mask_index = [1]*16000 + [0]*(22470-16000)
test_mask_index = [0]*16000 + [1]*(22470-16000)

# Create the dataset and assign the train and test masks
dataset = Data(x=x, y=y, edge_index=edge_index)
dataset.train_mask = torch.Tensor(train_mask_index).bool()
dataset.test_mask = torch.Tensor(test_mask_index).bool()

# check if there are any errors in the dataset
dataset.validate(raise_on_error=True)


