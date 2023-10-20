import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data():
    # Load the data
    data_dict = np.load('recognition/FacebookGCN_s4764408/facebook.npz', allow_pickle=True)
    edges = data_dict['edges']

    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(data_dict['features'])
    target = data_dict['target']

    # Convert to torch tensors
    edge_index = torch.tensor(edges, dtype=torch.long)
    x = torch.tensor(features_normalized, dtype=torch.float)
    y = torch.tensor(target, dtype=torch.long)

    # Create train/test split and convert them to boolean masks.
    train_indices, test_indices = train_test_split(range(len(y)), test_size=0.3, random_state=42)
    train_mask = torch.zeros(len(y), dtype=torch.bool)
    train_mask[train_indices] = True
    test_mask = torch.zeros(len(y), dtype=torch.bool)
    test_mask[test_indices] = True

    # Create torch_geometric data
    graph_data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y, train_mask=train_mask, test_mask=test_mask)
    
    return graph_data
