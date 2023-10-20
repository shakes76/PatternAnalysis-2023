
import torch
import torch.nn as nn
import torch.nn.functional as F

"""A single layer of a Graph Convolutional Network that aggregates information from neighboring 
    nodes and applies a linear transformation followed by a ReLU activation."""
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        
    def forward(self, x, adj):
        h = torch.mm(adj, x)
        h = self.fc(h)
        return F.relu(h)



"""A two-layer Graph Convolutional Network model that processes node features using adjacency information, applies
    dropout for regularization, and outputs log softmax values, while also providing an option to retrieve node embeddings from the first layer."""
class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(GCN, self).__init__()
        self.layer1 = GCNLayer(in_features, hidden_features)
        self.layer2 = GCNLayer(hidden_features, out_features)
        
    def forward(self, x, adj):
         # Store embeddings from the first layer (used for TSNE visualisation)
        self.embeddings = self.layer1(x, adj) 

        h = F.dropout(self.embeddings, p=0.5, training=self.training)
        h = self.layer2(h, adj)
        return F.log_softmax(h, dim=1)
    
    #  Retrieves the node embeddings produced after the first convolutional layer. (Used for TSNE visualisation)
    def get_embeddings(self):
        return self.embeddings