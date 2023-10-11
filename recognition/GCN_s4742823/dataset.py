import numpy as np
import torch
from torch_geometric.data import Data

# Directory for data
data_dir = 'data/'

def load_data(filepath: str = data_dir + 'facebook.npz') -> (Data, Data, Data):
    # Load the data
    data = np.load(filepath)
    edges = data['edges']
    features = data['features']
    target = data['target']

    edges_coo = torch.tensor(edges, dtype=torch.int64).t().contiguous()
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(target, dtype=torch.int64)

    test_size = 0.2  # 20% of the data for testing
    val_size = 0.1   # 10% of the data for validation

    num_nodes = x.size(0)
    num_test = int(num_nodes * test_size) # Calculate number of test nodes
    num_val = int(num_nodes * val_size) # Calculate number of validation nodes
    num_train = num_nodes - num_test - num_val # Calculate number of train nodes

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

    # Create the final Data object to be used for training
    data = Data(x=x, edge_index=edges_coo, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data, features