# Source code of the component for the model
# Thomas Bennion s4696627
import dgl
import torch as pt
import numpy as np
import predict

'''
GCN(pt.nn.Module):
Class defining the GCN model.

It is a subclass of pytorch's 'nn.Module'.

The graph convolution layer aggregates information from the nodes neighbours
and updates the node's feature representation.

The graph convolution operation is based on an aggregation of the 
features of neighbouring nodes. These features tend to have weights applied to 
them during training.

The final layer is a linear classifier that maps the output of the last 
GCN layer to class scores.
'''
class GCN(pt.nn.Module):

    '''
    init(in_feats, hidden_size, num_classes, num_layers):
    initialiser of the GCN model.

    Uses ReLU activation functions as it is a non-linear activation function,
    which allows the network to model more complex, non-linear relationships 
    in the dataset.

    input_feats: dimensionality of the features
    hidden_size: dimensionality of the hidden layers in the GCN
    num_classes: number of classes in the dataset
    num_layers: number of GCN layers
    '''
    def __init__(self, input_feats, hidden_size, num_classes, num_layers):
        super(GCN, self).__init__()
        self.layers = pt.nn.ModuleList()
        self.layers.append(dgl.nn.GraphConv(input_feats, hidden_size, activation=pt.nn.ReLU()))
        for _ in range(1, num_layers):
            self.layers.append(dgl.nn.GraphConv(hidden_size, hidden_size, activation=pt.nn.ReLU()))
        self.classifier = pt.nn.Linear(hidden_size, num_classes)

    '''
    forward(graph, inputs):
    Forward function for the GCN model.
    
    Applies the GCN layers to the feature vectors before passing the 
    final node representations through the linear classifier to 
    obtain the class scores.

    graph: dgl graph used to represent the structure of the graph
    inputs: Feature vectors associated with each node in the graph

    @returns: Class scores
    '''
    def forward(self, graph, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(graph, x)
        return self.classifier(x)