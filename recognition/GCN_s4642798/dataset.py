import numpy as np
import torch

# Load data file
facebook_data = np.load("recognition/GCN_s4642798/facebook.npz")

# Extract individual numpy arrays
edges = facebook_data["edges"]
features = facebook_data["features"]
target = facebook_data["target"]

# Convert Numpy Arrays to Tensors
edges = torch.Tensor(edges)
features = torch.Tensor(features)
target = torch.Tensor(target)

# Print Shape of Tensors
print("Edges Shapes: {}".format(edges.shape))
print("Features Shapes: {}".format(features.shape))
print("Target Shapes: {}".format(target.shape))
