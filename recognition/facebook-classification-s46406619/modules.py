import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv

# optimised GCN model for the given dt
class GCN(torch.nn.Module):
    def __init__(self, train_split=None, test_split=None):
        super().__init__()
        self.train_split = train_split
        self.test_split = test_split
        self.gcn1 = GCNConv(128, 64)
        self.gcn2 = GCNConv(64, 32)
        self.gcn3 = GCNConv(32, 16)
        self.linear = Linear(16, 4)

    def forward(self, x, edges):
        x = self.gcn1(x, edges).relu()
        x = self.gcn2(x, edges).relu()
        x = self.gcn3(x, edges).relu()
        z = self.linear(x)
        return x, z
    
    def accuracy(self, true, predict):
        return (true == predict).sum() / len(predict)