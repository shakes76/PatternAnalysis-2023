import torch
import numpy as np
from torch.nn import Linear
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn = GCNConv(128, 3)
        self.linear = Linear(3, 4)

    def forward(self, x, edges):
        h = self.gcn(x, edges).relu()
        z = self.linear(h)
        return h, z
    
    def accuracy(self, true, predict):
        return (true == predict).sum() / len(predict)
    
    def train(self, data, criterion, optimizer, num_epoch):
        for epoch in range(num_epoch):
            optimizer.zero_grad() # clear gradients
            h, z = self(data.X, data.edges) # forward pass
            
            # disregard all test labels from criterion
            y_train = data.target
            for i in range(len(z)):
                if data.split_indices[i] == 1: # remove all test elements
                    z = torch.cat([z[0:i], z[i+1:]])
                    y_train = torch.cat([y_train[0:i], y_train[i+1:]])

            loss = criterion(z, y_train) # calculate loss
            acc = self.accuracy(y_train, z.argmax(dim=1)) # calculate accuracy
            loss.backward() # compute gradients
            optimizer.step() # tune parameters
            
            # print metrics every 10 epochs
            if epoch % 10 == 0:
                print(f'Epoch {epoch:>3} | Loss: {loss:.2f} | Acc: {acc*100:.2f}%')