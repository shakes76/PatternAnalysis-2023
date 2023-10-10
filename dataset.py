import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit


def load_data():
    '''
    Loads the data
    '''
    features = torch.Tensor(np.load('facebook_large/features.npy'))
    edges = np.rot90(np.load('facebook_large/edges.npy'), 1)
    edges = torch.Tensor(edges.copy())
    targets = torch.Tensor(np.load('facebook_large/target.npy'))
    
    data = Data(x=features, edge_index=edges, edge_attr=None, y=targets)
    
    return data

def test_train():
    '''
    Creates training, validation and test masks
    '''
    data = load_data()
    split = RandomNodeSplit(num_val=0.2, num_test=0.1)
    
    return split(data)


def print_stats():
    '''
    Gather some statistics about the graph.
    '''
    data = test_train()
    
    print(data)
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Number of training nodes: {data.train_mask.sum()}')
    print(f'Number of validation nodes: {data.val_mask.sum()}')
    print(f'Number of test nodes: {data.test_mask.sum()}')
    print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')

# data = test_train()
# print(data[0])