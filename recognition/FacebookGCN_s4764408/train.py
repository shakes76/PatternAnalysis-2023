import torch
from modules import GCN
from dataset import load_data

# Load data
graph_data = load_data()

# Instantiate the model
model = GCN(num_features=128, hidden_channels=64, num_classes=len(set(graph_data.y)))

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    optimizer.zero_grad()
    out = model(graph_data.x, graph_data.edge_index)
    loss = criterion(out[graph_data.train_mask], graph_data.y[graph_data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test():
    model.eval()
    out = model(graph_data.x, graph_data.edge_index)
    pred = out.argmax(dim=1)
    correct = pred[graph_data.test_mask].eq(graph_data.y[graph_data.test_mask]).sum().item()
    acc = correct / graph_data.test_mask.sum().item()
    return acc, pred

if __name__ == "__main__":
    for epoch in range(200):
        loss = train()
        if epoch % 10 == 0:
            acc, _ = test()
            print(f"Epoch: {epoch}, Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")
    
    # Save model
    torch.save(model.state_dict(), "gcn_model.pth")
