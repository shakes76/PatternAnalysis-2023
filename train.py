import torch

from modules import GCN
from dataset import test_train


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = test_train()
model = GCN(hidden_channels=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    
    x = data.x.float()
    outputs = model(x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(outputs[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss

for epoch in range(1, 10):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

torch.save(model, 'GCN.pt')