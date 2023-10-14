import torch
import dataset
import modules
from plot import plot_accuracy,plot_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = modules.multi_GCN(
    dataset.sample_size, dataset.features_size, dataset.classes_size, 32
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

class LabelSmoothingCrossEntropy(torch.nn.Module):
    def __init__(self, eps=0.1):
        super().__init__()
        self.eps = eps

    def forward(self, output, target):
        # C is the number of classes
        C = output.size(1)
        # Distrubute the epsilon value among other class probabilities
        smoothed_labels = torch.full(size=(output.size()),
                                     fill_value=self.eps / (C - 1),
                                     device=output.device)
        # Assign the (1-eps) to the probability of the target class
        smoothed_labels.scatter_(1, target.data.unsqueeze(1), 1 - self.eps)
        # Calculate the negative log likelihood
        log_prob = torch.nn.functional.log_softmax(output, dim=1)
        loss = -log_prob * smoothed_labels
        loss = loss.sum(-1).mean()
        return loss

# using Label Smoothing Loss
criterion = LabelSmoothingCrossEntropy(eps=0.1)



criterion = LabelSmoothingCrossEntropy(eps=0.1).to(device)

# Ensure your data is on the correct device
X = dataset.X.to(device)
y = dataset.y.to(device)
adjacency_matrix = dataset.adjacency_matrix.to(device)
train_mask = dataset.train_mask.to(device)
val_mask = dataset.val_mask.to(device)
test_mask = dataset.test_mask.to(device)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(X, adjacency_matrix)  # Use tensors on GPU
    loss = criterion(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def test(mask):
    model.eval()
    # to disable gradient computation during inference
    with torch.no_grad():
        out = model(X, adjacency_matrix)  # use GPU tensors
        pred = out.argmax(dim=1)
        test_correct = pred[mask] == y[mask]  # ensure targets are on GPU
        test_acc = int(test_correct.sum()) / int(mask.sum())
    return test_acc


def validate():
    model.eval()
    out = model(X, adjacency_matrix)  # Use tensors on GPU
    loss = criterion(out[val_mask], y[val_mask])
    return loss.item()


# Training Loop
train_loss = []
val_loss = []
val_acc = []
train_acc = []

for epoch in range(1, 801):
    train_l = train()
    val_l = validate()

    train_loss.append(train_l)
    val_loss.append(val_l)
    val_acc.append(test(val_mask))
    train_acc.append(test(train_mask))
    print(f"Epoch: {epoch:03d}, Train Loss: {train_l:.4f}, Val Loss: {val_l:.4f}")
test_acc = test(test_mask)
print(f"Test Accuracy: {test_acc:.4f}")

plot_accuracy(train_acc, val_acc, "gcn_accuracy.png")
plot_loss(train_loss, val_loss, "gcn_loss.png")
torch.save(model.state_dict(), "best_model.pt")