# data loader for loading and preprocessing the data
import numpy as np
import dgl
import torch as pt
from dgl.data.utils import split_dataset

#Definitions
edges_dir = 'facebook/edges.npy'
features_dir = 'facebook/features.npy'
target_dir = 'facebook/target.npy'
train_ratio = 0.8

'''
Load and preprocess the data used in training and testing the algorithm
Loads the edges, features and target data into numpy arrays.
Creates a Deep Graph Library (DGL) to be used for the GCN,
This GCN will contain Features and Target information

@returns DGLGraph
'''
def load_data():
    edges_data = np.load(edges_dir)
    features_data = np.load(features_dir)
    target_data = np.load(target_dir)

    graph = dgl.graph((edges_data[:, 0], edges_data[:, 1]))

    graph.ndata['Features'] = pt.Tensor(features_data)
    graph.ndata['Target'] = pt.LongTensor(target_data)

    
    num_nodes = graph.number_of_nodes()
    train_nodes = int(train_ratio * num_nodes)
    train_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[:train_nodes] = True
    test_mask = ~train_mask

    labels = pt.zeros(num_nodes, dtype=pt.long)
    labels[train_mask] = 1

    graph.ndata['label'] = labels

    return graph
