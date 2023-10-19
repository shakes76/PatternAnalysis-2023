# Source code for training, validating, testing and saving the model
# Thomas Bennion s4696627
import torch as pt
import numpy as np
import predict
import dgl
import dataset
import train
import modules

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

        predict.evaluate(model, graph, test_mask, epoch, loss)
    
    predict.show_graph(graph, model)

'''
start():
starts the model
'''
def start():
    #Load and preprocess the data
    graph, train_mask, test_mask, num_features = dataset.load_data()

    # Initialize the GCN model
    model = modules.GCN(input_feats=128, hidden_size=64, num_classes=num_features, num_layers=2)

    #Train the model
    train.train_model(graph, train_mask, test_mask, model, 100, 0.01)

start()