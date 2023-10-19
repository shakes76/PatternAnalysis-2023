# GCN-Project
This project seeks to create a suitable multi-layer GCN model to carry out a semi supervised multi-class node classification using the [Facebook Large Page-Page Network dataset](https://snap.stanford.edu/data/facebook-large-page-page-network.html). To perform this a [Partially processed dataset](https://graphmining.ai/datasets/ptg/facebook.npz) was used where the features are in the form of 128 dim vectors.

The graph convolution layer aggregates information from the nodes neighbours
and updates the node's feature representation. The graph convolution operation is based on an aggregation of the features of neighbouring nodes. These features tend to have weights applied to them during training.

The final layer is a linear classifier that maps the output of the last 
GCN layer to class scores.

# Project Files
The implementation is split across four files, 'dataset.py', 'modules.py', 'train.py' and 'predict.py'.

## dataset.py
This file is used for loading and preprocessing the dataset. It contains one function 'load_data()' that loads and preprocesses the data into a Deep Graph Library (DGL) graph. It also creates two numpy arrays train_mask and test_mask, these state which values of the graph are for training and which are for testing. It also sends back information about the number of features in the model.

## modules.py
Contains the source code of the components of the model.

## train.py
This file is used training, validating, testing and saving the model

## predict.py
This file is used to show example usage of the trained model

# References
- 'dgl' library, found at: https://docs.dgl.ai/
