from torch import nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GCN(nn.Module):
    """ GCN model. Runs graph convolution until the final layer, which is a linear layer, 
     acts as a classifier from the latent space. """
    def __init__(self, in_channels, n_classes, hidden_layers = []):
        super().__init__()
        if hidden_layers:
            self.conv_layers = nn.ModuleList(
                [
                    GCNConv(in_channels, hidden_layers[0]),
                ]
            )

            for i in range(len(hidden_layers)-1):
                self.conv_layers.append(
                    GCNConv(hidden_layers[i], hidden_layers[i+1])
                )
            self.linear = nn.Linear(hidden_layers[-1], n_classes)
        else:
            self.conv_layers = GCNConv(in_channels, in_channels)
            self.linear = nn.Linear(in_channels, n_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x, edge_index)
            x = F.relu(x)
            # x = F.dropout(x, p=0.05)
        x = self.linear(x)
        return x
    
    def embeddings(self, data):
        """ Generate node embeddings of a graph """
        x, edge_index = data.x, data.edge_index
        for i in range(len(self.conv_layers)-1):
            x = self.conv_layers[i](x, edge_index)
            x = F.relu(x)
        x = self.conv_layers[-1](x, edge_index)
        return x   