import dataset
import modules
import torch


model = modules.GCN(
    dataset.sample_size, dataset.number_features, dataset.number_classes, 16
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()


def train():
    model.train()
    optimizer.zero_grad()
    out = model(dataset.X, dataset.edges_sparse)
    loss = criterion(out[dataset.train_mask], dataset.y[dataset.train_mask])
    loss.backward()
    optimizer.step()
    return loss


def test():
    model.eval()
    out = model(dataset.X, dataset.edges_sparse)
    pred = out.argmax(dim=1)
    test_correct = pred[dataset.test_mask] == dataset.y[dataset.test_mask]
    test_acc = int(test_correct.sum()) / int(dataset.test_mask.sum())
    return test_acc


for epoch in range(1, 101):
    loss = train()
    print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")

test_acc = test()
print(f"Test Accuracy: {test_acc:.4f}")
