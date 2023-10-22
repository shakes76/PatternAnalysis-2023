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

Code partially inspired by the dgl node classification tutorials 
found at:
https://docs.dgl.ai/tutorials/blitz/1_introduction.html#sphx-glr-tutorials-blitz-1-introduction-py


graph: dgl.Graph variable containing the preprocessed data
train_mask: numpy array stating which values are for the training dataset
test_mask: numpy array stating which values are for the testing dataset
model: GCN model
epochs: number of epochs of the model
learning_rate: learning rate of the model
'''
def train_model(graph, train_mask, test_mask, model, epochs, learning_rate):
    # Define loss function and optimizer
    loss_function = pt.nn.CrossEntropyLoss()
    optimizer = pt.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        logits = model(graph, graph.ndata['Features'])
        loss = loss_function(logits[train_mask], graph.ndata['Target'][train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluate the model on the test set
        model.eval()
        with pt.no_grad():
            logits = model(graph, graph.ndata['Features'])
            predictions = logits.argmax(1)
            accuracy = ((predictions[test_mask] == graph.ndata['Target'][test_mask]).float()).mean()

        predict.print_results(accuracy, epoch, loss)
    
    predict.show_graph(graph, model)

'''
start():
starts the model
'''
def start():
    #Load and preprocess the data
    graph, train_mask, test_mask, num_features = dataset.load_data()

    # Initialize the GCN model
    model = modules.GCN(input_feats=128, num_classes=num_features)

    #Train the model
    train.train_model(graph, train_mask, test_mask, model, 60, 0.01)

if __name__ == "__main__":
    start()