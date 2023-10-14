import torch
import dataset
import modules
from plot import plot_accuracy, plot_loss


def train_one_epoch(model, optimizer, criterion, X, adjacency_matrix, y, train_mask):
    model.train()
    optimizer.zero_grad()
    out = model(X, adjacency_matrix)
    loss = criterion(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate_model(model, X, adjacency_matrix, y, mask, criterion=None):
    model.eval()
    with torch.no_grad():
        out = model(X, adjacency_matrix)
        pred = out.argmax(dim=1)
        acc = int((pred[mask] == y[mask]).sum()) / int(mask.sum())
        if criterion is not None:
            loss = criterion(out[mask], y[mask]).item()
            return acc, loss
        return acc


def train_model(model, optimizer, criterion, X, adjacency_matrix, y, masks, num_epochs=800):
    train_mask, val_mask, _ = masks
    train_losses, val_losses, val_accuracies, train_accuracies = [], [], [], []

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, optimizer, criterion, X, adjacency_matrix, y, train_mask)
        val_acc, val_loss = evaluate_model(model, X, adjacency_matrix, y, val_mask, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        train_accuracies.append(evaluate_model(model, X, adjacency_matrix, y, train_mask))

        print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return train_losses, val_losses, train_accuracies, val_accuracies


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Ensure your data is on the correct device
    X = dataset.X.to(device)
    y = dataset.y.to(device)
    adjacency_matrix = dataset.adjacency_matrix.to(device)
    train_mask = dataset.train_mask.to(device)
    val_mask = dataset.val_mask.to(device)
    test_mask = dataset.test_mask.to(device)

    model = modules.multi_GCN(
        dataset.sample_size, dataset.features_size, dataset.classes_size, 32
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = LabelSmoothingCrossEntropy(eps=0.1).to(device)

    masks = (train_mask, val_mask, test_mask)

    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, optimizer, criterion, X, adjacency_matrix, y, masks, num_epochs=800
    )

    test_acc = evaluate_model(model, X, adjacency_matrix, y, test_mask)
    print(f"Test Accuracy: {test_acc:.4f}")

    plot_accuracy(train_accuracies, val_accuracies, "gcn_accuracy.png")
    plot_loss(train_losses, val_losses, "gcn_loss.png")

    torch.save(model.state_dict(), "best_model.pt")


if __name__ == "__main__":
    main()
