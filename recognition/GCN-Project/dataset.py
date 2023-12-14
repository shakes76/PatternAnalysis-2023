# data loader for loading and preprocessing the data
# Thomas Bennion s4696627
import numpy as np
import dgl
import torch as pt

'''
load_data():
Load and preprocess the data used in training and testing the algorithm

Loads the edges, features and target data into numpy arrays.

Creates a Deep Graph Library (DGL) graph to be used for the GCN,
This graph contains Edges, Features and Target information.

Creates two numpy arrays train_mask and test_mask, these state
which values of the graph are for training and which are for testing.

@returns: DGL graph, train_mask, test_mask, number of features
'''
def load_data():
    #Definitions
    edges_dir = 'facebook/edges.npy'
    features_dir = 'facebook/features.npy'
    target_dir = 'facebook/target.npy'
    train_ratio = 0.8
    
    #Load data from files
    edges_data = np.load(edges_dir)
    features_data = np.load(features_dir)
    target_data = np.load(target_dir)

    #Create the dgl graph
    graph = dgl.graph((edges_data[:, 0], edges_data[:, 1]))

    graph.ndata['Features'] = pt.Tensor(features_data)
    graph.ndata['Target'] = pt.LongTensor(target_data)

    #Creates the training and testing arrays
    num_nodes = graph.number_of_nodes()
    train_nodes = int(train_ratio * num_nodes)
    train_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[:train_nodes] = True
    test_mask = ~train_mask

    return graph, train_mask, test_mask, len(features_data)
