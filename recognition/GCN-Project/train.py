# Source code for training, validating, testing and saving the model
import dgl.nn as dglnn
import torch.nn as nn
import torch.optim as optim
import torch as pt
import numpy as np
import dgl

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes, num_layers):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.GraphConv(in_feats, hidden_size, activation=nn.ReLU()))
        for _ in range(1, num_layers):
            self.layers.append(dglnn.GraphConv(hidden_size, hidden_size, activation=nn.ReLU()))
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, graph, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(graph, x)
        return self.classifier(x)


def train_model(graph, feature_dim, num_classes, train_mask, test_mask):
    # Define hyperparameters
    num_epochs = 100
    learning_rate = 0.01

    # Initialize the model
    model = GCN(in_feats=feature_dim, hidden_size=64, num_classes=num_classes, num_layers=2)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
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
            accuracy = (predictions[test_mask] == graph.ndata['Target'][test_mask]).float().mean()

        print(f'Epoch {epoch}: Loss {loss.item()}, Test Accuracy {accuracy.item()}')
        