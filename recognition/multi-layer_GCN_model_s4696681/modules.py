import torch
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch.nn as nn
'''updated GCN layer trying to incorporate techniques learnt from model expo CON session'''
class GCNLayer(Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight) 

    def forward(self, features, adj, active=True):
        support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        if active:
            output = F.relu(output)
        return output

'''GCN represents the entire Graph Convolutional Network consisting of two GCNLayer'''
class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(GCN, self).__init__()
        self.layer1 = GCNLayer(in_features, hidden_features)
        self.layer2 = GCNLayer(hidden_features, out_features)
        
    def forward(self, x, adj):
        # Use the first GNNLayer with activation function (ReLU)
        self.embeddings = self.layer1(x, adj, active=True) 
        h = F.dropout(self.embeddings, p=0.5, training=self.training)
        # Use the second GNNLayer without activation function
        h = self.layer2(h, adj, active=False) 
        return F.log_softmax(h, dim=1)
    
    def get_embeddings(self):
        return self.embeddings
