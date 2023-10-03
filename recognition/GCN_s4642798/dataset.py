import numpy as np
import torch

# Load data file
facebook_data = np.load("facebook.npz")

# Extract individual numpy arrays
edges = facebook_data["edges"]
features = facebook_data["features"]
target = facebook_data["target"]

# Convert Numpy Arrays to Tensors
edges = torch.Tensor(edges)
X = torch.Tensor(features)
y = torch.Tensor(target)

# Cast tensors that contain integer values to integer type
edges = edges.to(torch.int64)
y = y.to(torch.int64)

# Print Shape of Tensors
print("Edges Shape: {}".format(edges.shape))
print("X Shape: {}".format(X.shape))
print("Y Shape: {}".format(y.shape))

# Save and output data information
sample_size = int(X.size(0))
number_features = int(X.size(1))
number_classes = len(torch.unique(y))
print("Sample Size: {}".format(sample_size))
print("Number of Features: {}".format(number_features))
print("Number of Classes: {}".format(number_classes))

# adding self loops to the tensor
self_loops = torch.eye(sample_size)
sparse_self_loops = torch.nonzero(self_loops, as_tuple=False)
condition = edges[:, 0] != edges[:, 1]
edges = edges[condition]
edges = torch.cat((edges, sparse_self_loops), 0)

# converting edges tensor to sparse tensor
value = torch.ones(edges.size(0))
edges_sparse = torch.sparse_coo_tensor(
    edges.t(), value, torch.Size([sample_size, sample_size])
)

# Generate training and testing mask
train_split = 0.7
val_split = 0.15
train_size = int(train_split * sample_size)
val_size = int(val_split * sample_size)

rand_indicies = torch.randperm(sample_size)
train_indicies = rand_indicies[:train_size]
val_indicies = rand_indicies[train_size : (train_size + val_size)]
test_indicies = rand_indicies[(train_size + val_size) :]

train_mask = torch.zeros(sample_size, dtype=torch.bool)
train_mask[train_indicies] = True

val_mask = torch.zeros(sample_size, dtype=torch.bool)
val_mask[val_indicies] = True

test_mask = torch.zeros(sample_size, dtype=torch.bool)
test_mask[test_indicies] = True

# Outputing size of training and testing set
print("Size of Training Set: {}".format(torch.sum(train_mask).item()))
print("Size of Test Set: {}".format(torch.sum(test_mask).item()))
