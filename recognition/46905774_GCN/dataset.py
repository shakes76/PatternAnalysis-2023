import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the data and convert them directly to PyTorch tensors with specified types
facebook_data = np.load("/content/drive/MyDrive/gcn/draft/facebook.npz")

edges = torch.tensor(facebook_data["edges"], dtype=torch.long)
X = torch.tensor(facebook_data["features"], dtype=torch.float32)
y = torch.tensor(facebook_data["target"], dtype=torch.long)

# Display shapes and feature information
print(f"Edges Shape: {edges.shape}")
print(f"X Shape: {X.shape}")
print(f"Y Shape: {y.shape}")
print(f"Sample Size: {X.size(0)}")
print(f"Features size: {X.size(1)}")
print(f"Classes size: {len(torch.unique(y))}")

features_size = X.size(1)  # number of input features per node
classes_size = len(torch.unique(y))  # number of output classes

# Add self-loops
self_loops = torch.arange(X.size(0)).view(-1, 1).repeat(1, 2)
edges_with_loops = torch.vstack([edges, self_loops])

# Construct a sparse adjacency matrix
adjacency_matrix = torch.sparse_coo_tensor(
    edges_with_loops.t(), torch.ones(edges_with_loops.size(0)),
    (X.size(0), X.size(0))
).to(device)

# Split indices for training, validation, and test sets
sample_size = X.size(0)
indices = torch.randperm(sample_size)
train_end = int(0.7 * sample_size)
val_end = train_end + int(0.15 * sample_size)

train_indices = indices[:train_end]
val_indices = indices[train_end:val_end]
test_indices = indices[val_end:]

# Create masks for training, validation, and test sets
train_mask = torch.zeros(sample_size, dtype=torch.bool).scatter_(0, train_indices, True)
val_mask = torch.zeros(sample_size, dtype=torch.bool).scatter_(0, val_indices, True)
test_mask = torch.zeros(sample_size, dtype=torch.bool).scatter_(0, test_indices, True)

# Also make sure other data tensors are moved to the desired device
X = X.to(device)
y = y.to(device)
train_mask = train_mask.to(device)
val_mask = val_mask.to(device)
test_mask = test_mask.to(device)