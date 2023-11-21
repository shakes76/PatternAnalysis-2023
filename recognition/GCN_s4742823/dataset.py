'''
This file loads the data from the path "data/facebook.npz" (or the filepath given to load_data()) 
and prepares it for training or testing. It also contains the GCNData class, which stores all of 
the information regarding the data.
'''
import numpy as np
import torch
import scipy.sparse as sp
from utils import SEED

# Set seed for reproducibility.
np.random.seed(SEED)

# Directory for data
data_dir = 'data/'

# Data object to store all properties related to the data.
class GCNData:
    def __init__(self, x, y, train_mask, val_mask, test_mask, adj, features):
        self.x = x
        self.y = y
        # Masks used for splitting data.
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        # Adjacency matrix for connections.
        self.adj = adj
        self.features=features

    # This method allows us to move the GCNData object to GPU.
    def to(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.train_mask = self.train_mask.to(device)
        self.val_mask = self.val_mask.to(device)
        self.test_mask = self.test_mask.to(device)
        self.adj = self.adj.to(device)

        return self

def load_data(filepath: str = data_dir + 'facebook.npz', test_size=0.2, val_size=0.1) -> GCNData:
    # Load the data
    data = np.load(filepath)
    edges = data['edges']
    features = data['features']
    target = data['target']

    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(target, dtype=torch.int64)

    num_nodes = x.size(0)
    num_test = int(num_nodes * test_size)  # Calculate number of test nodes
    num_val = int(num_nodes * val_size)  # Calculate number of validation nodes
    num_train = num_nodes - num_test - num_val  # Calculate number of train nodes

    # Generate list of node indices, then shuffle them.
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)

    # Slice node indices into train, validation and test using calculated numbers of each
    train_idx, val_idx, test_idx = indices[:num_train], indices[num_train:num_train+num_val], indices[-num_test:]

    # Create masks for train, validation, and test sets
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = 1
    val_mask[val_idx] = 1
    test_mask[test_idx] = 1

    # Create an adjacency matrix from the edge information
    adj = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(num_nodes, num_nodes))
    adj = torch.Tensor(adj.todense())

    # Create the final Data object to be used for training
    data = GCNData(x=x, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, adj=adj, features=features)

    return data