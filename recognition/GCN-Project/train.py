# Source code for training, validating, testing and saving the model
# Thomas Bennion s4696627
import torch as pt
import numpy as np
import predict
import dgl

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

'''
train_model(graph, train_mask, test_mask, model, epochs, learning_rate):
Function to train the GCN model.

Uses a Cross-Entropy Loss function to measure the dissimilarity 
between the predicted class probabilities and the true class labels.

Uses the Adam optimiser as it is the most versatile, and performs well enough
for this problem.

graph: dgl.Graph variable containing the preprocessed data
train_mask: numpy array stating which values are for the training dataset
test_mask: numpy array stating which values are for the testing dataset
model: GCN model
epochs: number of epochs of the model
learning_rate: learning rate of the model
'''
def train_model(graph, train_mask, test_mask, model, epochs, learning_rate):
    # Define loss function and optimizer
    criterion = pt.nn.CrossEntropyLoss()
    optimizer = pt.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        logits = model(graph, graph.ndata['Features'])
        loss = criterion(logits[train_mask], graph.ndata['Target'][train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluate the model on the test set
        model.eval()
        with pt.no_grad():
            logits = model(graph, graph.ndata['Features'])
            predictions = logits.argmax(1)
            accuracy = ((predictions[test_mask] == graph.ndata['Target'][test_mask]).float()).mean()

        #print(f'Epoch {epoch}: Loss {loss.item()}, Test Accuracy {accuracy.item()}')
    
    predict.show_graph(graph, model)