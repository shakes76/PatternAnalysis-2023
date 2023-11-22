"""
Author: William Barker
SN: 4642798
This script is designed to handle the loading, preprocessing, 
and splitting of data from the facebook.npz file.
"""
import numpy as np
import torch


def load_data(filename):
    """
    Function to load data from .npz file and convert to tensors
    """
    # Load data file
    facebook_data = np.load(filename)

    # Extract individual numpy arrays
    edges = facebook_data["edges"]
    features = facebook_data["features"]
    target = facebook_data["target"]

    # Convert Numpy Arrays to Tensors
    edges = torch.Tensor(edges)
    X = torch.Tensor(features)
    y = torch.Tensor(target)

    # Print Shape of Tensors
    print("Edges Shape: {}".format(edges.shape))
    print("X Shape: {}".format(X.shape))
    print("Y Shape: {}".format(y.shape))

    return edges, X, y


def generate_data_info(X, y):
    """
    Function to determine sample size, number of features and number
    of classes in the data
    """
    # Save and output data information
    sample_size = int(X.size(0))
    number_features = int(X.size(1))
    number_classes = len(torch.unique(y))
    print("Sample Size: {}".format(sample_size))
    print("Number of Features: {}".format(number_features))
    print("Number of Classes: {}".format(number_classes))
    return sample_size, number_features, number_classes


def generate_data_split(train_split, val_split, sample_size):
    """
    Function to generate masks for the training, testing and validations sets
    """
    # Calculate the sizes of the training and validation set
    train_size = int(train_split * sample_size)
    val_size = int(val_split * sample_size)

    # Generate random indices for shuffling the data
    rand_indicies = torch.randperm(sample_size)

    # Create masks for the training, validation, and testing
    # sets based on the random indices
    train_indicies = rand_indicies[:train_size]
    train_mask = torch.zeros(sample_size, dtype=torch.bool)
    train_mask[train_indicies] = True

    val_indicies = rand_indicies[train_size : (train_size + val_size)]
    val_mask = torch.zeros(sample_size, dtype=torch.bool)
    val_mask[val_indicies] = True

    test_indicies = rand_indicies[(train_size + val_size) :]
    test_mask = torch.zeros(sample_size, dtype=torch.bool)
    test_mask[test_indicies] = True

    # Outputing size of training and testing set
    print("Size of Training Set: {}".format(torch.sum(train_mask).item()))
    print("Size of Validation Set: {}".format(torch.sum(val_mask).item()))
    print("Size of Test Set: {}".format(torch.sum(test_mask).item()))

    return train_mask, val_mask, test_mask


def preprocess_data(edges, X, y, sample_size):
    """
    Function to add self loops to adjaceny matrix and convert to sparse array.
    """
    # Cast tensors that contain integer values to integer type
    edges = edges.to(torch.int64)
    y = y.to(torch.int64)

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

    return edges, edges_sparse, X, y


edges, X, y = load_data("facebook.npz")
sample_size, number_features, number_classes = generate_data_info(X, y)
train_mask, val_mask, test_mask = generate_data_split(0.7, 0.15, sample_size)
edges, edges_sparse, X, y = preprocess_data(edges, X, y, sample_size)
