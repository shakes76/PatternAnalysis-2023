# Semi-supervised Multi-class Node Classification via GCN
This Graph Convolutional Model is used to perform semi-supervised multi-class node classification. The nodes in question are Facebook pages, the edges are mutual likes between these pages and the features are descriptions of the pages created by their owners. The classes are politicians, governmental organisations, television shows and companies.

## Data
This model uses a pre-processed dataset where features are in the form of 128 dim vectors. This was provided by the University of Queensland.

The data is loaded from a .npz file using numpy.load(), and the "edges", "features" and "targets" are extracted. The edges are stored as a 171,002 * 2 matrix, representing rows of connected nodes. The stored as a 22,470 * 128 matrix, meaning there are 22,470 nodes, each with 128 features. Hence the targets are represented with a one-dimensional list of 22,470 integers which correspond to the four classes. The features and targets are loaded into tensors as x and y respectively. Random masks for the test, train and validation data splits are calculated. An adjacency matrix is generated from the edge information to be used in the graph convolution. Finally, all of the information is combined into a single GCNData object to be used during the training.


## About the Model

### Architecture
The model consists of two sets of graph convolutional and batch normalisation layers. The first set is followed by a ReLU activation function and a dropout layer.

Graph convolution is...

### Hyperparameters and Functions

## Results and Figures

## Dependencies
Required packages are as follows:
- torch
- sklearn
- scipy
- matplotlib

## References
https://snap.stanford.edu/data/facebook-large-page-page-network.html
https://github.com/gayanku/SCGC
