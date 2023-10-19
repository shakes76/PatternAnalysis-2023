# data loader for loading and preprocessing the data
import numpy as np

#Definitions
edges_dir = 'facebook/edges.npy'
features_dir = 'facebook/features.npy'
target_dir = 'facebook/target.npy'
vector_dim = 128

def load_data():
    edges_data = np.load(edges_dir)
    features_data = np.load(features_dir)
    target_data = np.load(target_dir)
    return edges_data, features_data, target_data
