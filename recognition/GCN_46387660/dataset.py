import torch
import numpy as np
from torch_geometric.data import Data

# Load dataset from npz file
data = np.load('facebook.npz')
       
# allocate x, y and edge_index tensors
x = torch.from_numpy(data['features']).to(torch.float)
y = torch.from_numpy(data['target']).to(torch.long)
edge_index = torch.from_numpy(data['edges']).to(torch.long)
# changes the edge index to a list of index tuples
edge_index = edge_index.t().contiguous()

# create dataset
data = Data(x=x, y=y, edge_index=edge_index)

# check if there are any errors in the dataset
data.validate(raise_on_error=True)

# shuffle. then split data into train and test



