import dataset
import modules
import torch
import numpy as np
import matplotlib.pyplot as plt

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


def test(mask):
    model.eval()
    out = model(dataset.X, dataset.edges_sparse)
    pred = out.argmax(dim=1)
    test_correct = pred[mask] == dataset.y[mask]
    test_acc = int(test_correct.sum()) / int(mask.sum())
    return test_acc


val_acc = []
train_acc = []

for epoch in range(1, 101):
    loss = train()
    val_acc.append(test(dataset.val_mask))
    train_acc.append(test(dataset.train_mask))
    print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")

test_acc = test(dataset.test_mask)
print(f"Test Accuracy: {test_acc:.4f}")

plt.figure(figsize=(12, 8))
plt.plot(np.arange(1, len(val_acc) + 1), val_acc, label="Validation Accuracy")
plt.plot(np.arange(1, len(train_acc) + 1), train_acc, label="Training Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accurarcy")
plt.title("Training and Validation Accuracy")
plt.legend(loc="lower right", fontsize="x-large")
plt.savefig("gcn_loss.png")
plt.show()

torch.save(model.state_dict(), "best_model.pt")
