import torch
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
            loss = criterion(z, data.y) # calculate loss
            acc = self.accuracy(data.y, z.argmax(dim=1)) # calculate accuracy
            loss.backward() # compute gradients
            optimizer.step() # tune parameters
            
            # print metrics every 10 epochs
            if epoch % 10 == 0:
                print(f'Epoch {epoch:>3} | Loss: {loss:.2f} | Acc: {acc*100:.2f}%')