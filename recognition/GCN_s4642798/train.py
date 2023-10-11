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
    train_loss = criterion(out[dataset.train_mask], dataset.y[dataset.train_mask])
    train_loss.backward()
    optimizer.step()
    val_loss = criterion(out[dataset.val_mask], dataset.y[dataset.val_mask])
    return (train_loss, val_loss)


def test(mask):
    model.eval()
    out = model(dataset.X, dataset.edges_sparse)
    pred = out.argmax(dim=1)
    test_correct = pred[mask] == dataset.y[mask]
    test_acc = int(test_correct.sum()) / int(mask.sum())
    return test_acc


val_acc = []
train_acc = []
val_loss_all = []
train_loss_all = []

for epoch in range(1, 101):
    train_loss, val_loss = train()
    val_acc.append(test(dataset.val_mask))
    train_acc.append(test(dataset.train_mask))
    val_loss_all.append(float(val_loss))
    train_loss_all.append(float(train_loss))
    print(f"Epoch: {epoch:03d}, Loss: {train_loss:.4f}")

test_acc = test(dataset.test_mask)
print(f"Test Accuracy: {test_acc:.4f}")

plt.figure(figsize=(12, 8))
plt.plot(np.arange(1, len(val_acc) + 1), val_acc, label="Validation Accuracy")
plt.plot(np.arange(1, len(train_acc) + 1), train_acc, label="Training Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accurarcy")
plt.title("Training and Validation Accuracy")
plt.legend(loc="lower right", fontsize="x-large")
plt.savefig("gcn_accuracy.png")

plt.figure(figsize=(12, 8))
plt.plot(np.arange(1, len(val_loss_all) + 1), val_loss_all, label="Validation Loss")
plt.plot(np.arange(1, len(train_loss_all) + 1), train_loss_all, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend(loc="lower right", fontsize="x-large")
plt.savefig("gcn_loss.png")

torch.save(model.state_dict(), "best_model.pt")
