import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

# Directories for data
train_dir = 'data/'

# Load the data
data = np.load(train_dir + 'facebook.npz')
edges = data['edges']
features = data['features']
target = data['target']

edges_coo = torch.tensor(edges, dtype=torch.int64).t().contiguous()
x = torch.tensor(features, dtype=torch.float32)
y = torch.tensor(target, dtype=torch.int64)

test_size = 0.2  # 20% of the data for testing
val_size = 0.1   # 10% of the data for validation

data = Data(x=x, edge_index=edges_coo, y=y)

# To ensure validation and testing don't overlap, we split data into train and temp, then temp into test and validation.
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=(test_size + val_size), random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=(val_size / (val_size + test_size)), random_state=42)

train_set = Data(x=x_train, edge_index=edges_coo, y=y_train)
val_set = Data(x=x_val, edge_index=edges_coo, y=y_val)
test_set = Data(x=x_test, edge_index=edges_coo, y=y_test)