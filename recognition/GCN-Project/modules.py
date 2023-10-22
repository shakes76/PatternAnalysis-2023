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

This class was partially inspired from Gayan Kulatilleke's lecture on GNN's 
available at: https://github.com/gayanku/SCGC/blob/main/models.py


This class was also partially inspired by the dgl node classification tutorials 
found at:
https://docs.dgl.ai/tutorials/blitz/1_introduction.html#sphx-glr-tutorials-blitz-1-introduction-py



'''
class GCN(pt.nn.Module):

    '''
    init(in_feats, hidden_size, num_classes, num_layers):
    initialiser of the GCN model.

    Uses ReLU activation functions as it is a non-linear activation function,
    which allows the network to model more complex, non-linear relationships 
    in the dataset.

    It was found in testing that increasing the dimensionality in deeper layers 
    allowed for an improved model. It was also found an initial layer of 64 was
    ideal for getting the model to have reasonable accuracy.

    input_feats: dimensionality of the features
    num_classes: number of classes in the dataset
    '''
    def __init__(self, input_feats, num_classes):
        super(GCN, self).__init__()
        self.layers = pt.nn.ModuleList()

        self.layers.append(dgl.nn.GraphConv(input_feats, 64, activation=pt.nn.ReLU()))
        self.layers.append(dgl.nn.GraphConv(64, 64, activation=pt.nn.ReLU()))
        self.layers.append(dgl.nn.GraphConv(64, 96, activation=pt.nn.ReLU()))
        self.layers.append(dgl.nn.GraphConv(96, 96, activation=pt.nn.ReLU()))

        self.classifier = pt.nn.Linear(96, num_classes)

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