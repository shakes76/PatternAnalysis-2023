from torch.nn import Linear
from torch_geometric.nn import GCNConv

from dataset import *
from train import *

class GCN(torch.nn.Module):
    def __init__(self, lr=0.02):
        super().__init__()
        self.gcn = GCNConv(128, 3)
        self.out = Linear(3, 4)

    def forward(self, x, edge_index):
        h = self.gcn(x, edge_index).relu()
        z = self.out(h)
        return h, z
    
    def accuracy(true, predict):
        return (true == predict).sum() / len(predict)
    
    def train(self, data, criterion, optimizer, num_epoch):
        for epoch in range(num_epoch):
            optimizer.zero_grad() # clear gradients
            h, z = self(data.X_train, data.edges) # forward pass
            loss = criterion(z, data.y_train) # calculate loss
            acc = self.accuracy(data.y_train, z.argmax(dim=1)) # calculate accuracy
            loss.backward() # compute gradients
            optimizer.step() # tune parameters
            
            # print metrics every 10 epochs
            if epoch % 10 == 0:
                print(f'Epoch {epoch:>3} | Loss: {loss:.2f} | Acc: {acc*100:.2f}%')

def run_training(lr=0.02, num_epoch=100):
    data = load_data()
    model = GCN()
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion=torch.nn.CrossEntropyLoss()

    model.train(data, criterion, optimizer, num_epoch)

run_training()