import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

class Dataset:
    """
    Class to handle the data import and placement in a data loader.
    """
    def __init__(self, file_loc, test_size = 0.2, device='cpu'):
        self.file_loc = file_loc
        self.device = device

        # Train test split proportion
        self.test_size = test_size 

        self.import_data()

    def import_data(self):
        """ Import data from npz file """
        raw_np = np.load(self.file_loc)
        edges = torch.tensor(raw_np['edges'].T).to(self.device)
        node_features = torch.tensor(raw_np['features']).to(self.device)
        target = torch.tensor(raw_np['target']).to(self.device)

        all_indices = np.arange(node_features.shape[0])
        train_indices = np.random.choice(all_indices, int(node_features.shape[0]*0.8), replace=False)

        train_mask = np.isin(all_indices, train_indices)
        train_mask = torch.tensor(train_mask)

        test_mask = ~np.isin(all_indices, train_indices)
        test_mask = torch.tensor(test_mask)

        self.graph = Data(
            x=node_features, 
            edge_index=edges, 
            y=target,
            train_mask = train_mask, 
            test_mask = test_mask
        ).contiguous().to(self.device)

    def data_loader(self, batchsize=32):
        """ Output data loader """
        self.graph_nl = NeighborLoader(
            data=self.graph, 
            num_neighbors=[-1],
            batch_size=128
        )
        return self.graph_nl