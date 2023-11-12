"""
Author: William Barker
SN: 4642798
This script executes the training of a Graph Convolutional Network (GCN) model on the 
provided dataset over a specified number of epochs, while also generating plots for 
accuracy and loss, and saving the trained model for future predictions.
"""
import dataset
import modules
import torch
import numpy as np
import matplotlib.pyplot as plt


def train_epoch():
    """
    Function to train single epoch of the GCN model
    """
    model.train()  # Set model to train mode
    optimizer.zero_grad()  # Clear previous calulated gradients
    out = model(dataset.X, dataset.edges_sparse)  # Single Forward Pass

    # Compute training loss and complete backward pass
    train_loss = criterion(out[dataset.train_mask], dataset.y[dataset.train_mask])
    train_loss.backward()

    optimizer.step()  # Update model paramters
    # Compute validation loss loss
    val_loss = criterion(out[dataset.val_mask], dataset.y[dataset.val_mask])
    return (train_loss, val_loss)


def test(mask):
    """
    Function to compute accuracy for a given set mask.
    """
    model.eval()  # Set to evaluation mode
    out = model(dataset.X, dataset.edges_sparse)  # Peform Forward Pass
    pred = out.argmax(dim=1)  # Obtain Predicted Class Lables
    test_correct = pred[mask] == dataset.y[mask]  # Indexes of correct predictions
    test_acc = int(test_correct.sum()) / int(mask.sum())  # Calculate accuracy
    return test_acc


def train(epochs, patience=5):
    """
    Function to train the GCN model for a specified number of epochs with early stopping
    """
    val_acc = []
    train_acc = []
    val_loss_all = []
    train_loss_all = []

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    # Loop over the specified number of epochs
    for epoch in range(1, epochs + 1):
        train_loss, val_loss = train_epoch()
        val_acc_value = test(dataset.val_mask)
        train_acc.append(test(dataset.train_mask))
        val_acc.append(val_acc_value)
        val_loss_all.append(float(val_loss))
        train_loss_all.append(float(train_loss))
        print(
            f"Epoch: {epoch:03d}, Loss: {train_loss:.4f}, Validation Accuracy: {val_acc_value:.4f}"
        )

        # Check for early stopping
        if val_acc_value > best_val_acc:
            best_val_acc = val_acc_value
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"Early stopping at epoch {epoch}. Best validation accuracy: {best_val_acc:.4f}"
            )
            break

    # Calculate accuracy on test set
    test_acc = test(dataset.test_mask)
    print(
        f"Test Accuracy: {test_acc:.4f} (Best validation accuracy at epoch {best_epoch})"
    )

    return val_acc, train_acc, val_loss_all, train_loss_all, test_acc


def plot_accuracy(val_acc, train_acc):
    """
    Function to plot the training and validaiton accuracy
    """
    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(1, len(val_acc) + 1), val_acc, label="Validation Accuracy")
    plt.plot(np.arange(1, len(train_acc) + 1), train_acc, label="Training Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accurarcy")
    plt.title("Training and Validation Accuracy")
    plt.legend(loc="lower right", fontsize="x-large")
    plt.savefig("plots/gcn_accuracy.png")


def plot_loss(val_loss_all, train_loss_all):
    """
    Function to plot the training and validaiton loss
    """
    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(1, len(val_loss_all) + 1), val_loss_all, label="Validation Loss")
    plt.plot(
        np.arange(1, len(train_loss_all) + 1), train_loss_all, label="Training Loss"
    )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend(loc="upper right", fontsize="x-large")
    plt.savefig("plots/gcn_loss.png")


# initilize model with specified parameters, ADAM optimizer and Cross entropy loss
model = modules.GCN(
    dataset.sample_size, dataset.number_features, dataset.number_classes, 16
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# Train model over 100 epochs with early stopping (patience=10)
val_acc, train_acc, val_loss_all, train_loss_all, test_acc = train(100, patience=5)
plot_accuracy(val_acc, train_acc)
plot_loss(val_loss_all, train_loss_all)

torch.save(model.state_dict(), "best_model.pt")  # Save model for later predictions
