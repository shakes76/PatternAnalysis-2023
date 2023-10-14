import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def load_data():
    '''
    Loads the data
    '''
    features = np.load('facebook_large/features.npy')
    features = torch.Tensor(np.load('facebook_large/features.npy'))
    edges = np.rot90(np.load('facebook_large/edges.npy'), 1)
    edges = torch.Tensor(edges.copy()).to(torch.int64)
    targets = torch.Tensor(np.load('facebook_large/target.npy')).to(torch.int64)
    
    data = Data(x=features, edge_index=edges, edge_attr=None, y=targets)
    
    return data

def test_train():
    '''
    Creates training, validation and test masks
    '''
    data = load_data()
    split = RandomNodeSplit(num_val=0.5, num_test=0.4)
    
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

def visualise():
    '''
    Visualise the data using TSNE embedding
    
    '''
    features = np.load('facebook_large/features.npy')
    targets = np.load('facebook_large/target.npy')
    
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)
    
    # Plotting stuff
    # Plot t-SNE embeddings with ground truth labels as colors
    plt.figure(figsize=(10, 8))
    
    # Define a color map for the labels
    labels = [0, 1, 2, 3]
    colors = plt.cm.rainbow(np.linspace(0, 1, len(labels)))
    
    for i, label in enumerate(labels):
        # Filter data points for each label
        mask = (targets == label)
        plt.scatter(tsne_results[mask, 0], tsne_results[mask, 1], label=f'Label {label}', color=colors[i])

    plt.title('TSNE Embeddings with Ground Truth Labels')
    plt.legend()
    plt.show()

# data = test_train()
# print_stats()
# visualise()