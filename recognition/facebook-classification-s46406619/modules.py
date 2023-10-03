from torch.nn import Linear
from torch_geometric.nn import GCNConv

from dataset import *
from train import *

X_train, y_train, X_test, y_test, edges = load_data()

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn = GCNConv(128, 3)
        self.out = Linear(3, 4)

    def forward(self, x, edge_index):
        h = self.gcn(x, edge_index).relu()
        z = self.out(h)
        return h, z

model = GCN()
print(model)