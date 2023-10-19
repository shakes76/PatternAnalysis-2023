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
The model uses a ReLU activation function after the first convolution layer.

For training, the optimizer used is Adam, and the loss function is Cross Entropy Loss.

The chosen hyperparameters are as follows:
- Number of epochs: 500
- Learning rate = 1e-2
- Dropout probability = 0.5
- Hidden dimensions = 64

## Figures and Results 
Below is a plot displaying the model's loss throughout training and including validation.

![The accuracy plot for both training and validation](./loss.png)

Below is a plot displaying the model's accuracy throughout training and including validation.

![The accuracy plot for both training and validation](./accuracy.png)

We can observe a small amount of overfitting, as the training loss is slightly lower than the validation loss (and the opposite is true of accuracy), however this is to be expected, and the difference is not significant.

Below is a t-SNE plot generated after training.

![A t-SNE plot generated after training.](./tsne_plot.png)

We can observe fairly distinct clusters in the t-SNE plot, suggesting the model has successfully learned meaningful representations of the data in lower dimensions.

The model's average accuracy is around 90-92%. This fits with the above t-SNE plot.

## Files and Usage
- `dataset.py` - Contains classes and functions for loading and preparing data for training / testing.
- `modules.py` - Contains the Model class and its supporting modules.
- `utils.py` - Sets the device to GPU where possible, and contains constants.
- `train.py` - Contains functions for training, testing and plotting. Can be run to train and test the model, and plot the results.
- `predict.py` - Loads the model if saved, and tests it, then plots the t-SNE results.

Note that the `SEEDS` constant in utils.py can be used to guarantee reproducibility. 

To train and test the model, and then plot the results, run `train.py`. To test an already existing model, run `test.py`.

Note that by default, `dataset.py` expects to read from a file `facebook.npz` in the `data` directory. `train.py` expects to load a model called `Facebook_GCN.pth`

## Dependencies
Required packages are as follows:
- torch
- sklearn
- scipy
- matplotlib

## References
https://snap.stanford.edu/data/facebook-large-page-page-network.html

https://github.com/gayanku/SCGC
