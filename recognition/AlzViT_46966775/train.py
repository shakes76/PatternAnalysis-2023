import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

from dataset import get_train_val_loaders, get_test_loader
from modules import ViT


def initialize_device():
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    return device


def initialize_model(device):
    print("Assigning model instance...")
    model = ViT(
        in_channels=1,
        patch_size=14,
        emb_size=1536,
        img_size=224,
        depth=10,
        n_classes=2,
    ).to(device)
    print("Model ready.")
    return model


def print_confusion_matrix(title, TP, TN, FP, FN):
    print(f"{title}:")
    print("-----------------------")
    print(f"|{'True AD (TP)':<25}|{TP:>5}|")
    print(f"|{'True NC (TN)':<25}|{TN:>5}|")
    print(f"|{'False AD (FP)':<25}|{FP:>5}|")
    print(f"|{'False NC (FN)':<25}|{FN:>5}|")
    print("-----------------------\n")


def train_and_validate(
    model, train_loader, val_loader, num_epochs, criterion, optimizer, scheduler
):
    best_val_accuracy = 0

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        total_train_loss, total_train_correct, total_samples = (
            0,
            0,
            len(train_loader.dataset),
        )
        batch_loss, batch_correct = 0, 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward Pass
            logits = model(images)
            loss = criterion(logits, labels)

            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log Info
            pred = logits.argmax(dim=1)
            correct = pred.eq(labels).sum().item()

            # For Batch Print
            batch_loss += loss.item()
            batch_correct += correct

            # For Training Log Values
            total_train_loss += loss.item()
            total_train_correct += correct

            if batch_idx % 10 == 0:
                avg_batch_loss = batch_loss / (10 * len(images))
                avg_batch_accuracy = batch_correct / (10 * len(images))
                print(
                    f"Epoch {epoch}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, 10 Batch Average Training Loss: {avg_batch_loss}, 10 Batch Average Training Accuracy: {avg_batch_accuracy}"
                )
                batch_loss = 0
                batch_correct = 0

        # Storing Training Epoch Performance
        train_losses.append(total_train_loss / total_samples)
        train_accuracies.append(total_train_correct / total_samples)

        # Validation Phase
        model.eval()
        total_loss, total_correct, total_samples = 0, 0, len(val_loader.dataset)

        # Counters for True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN)
        TP, TN, FP, FN = 0, 0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                pred = logits.argmax(dim=1)
                correct = pred.eq(labels).sum().item()
                TP += ((pred == 0) & (labels == 0)).sum().item()
                TN += ((pred == 1) & (labels == 1)).sum().item()
                FP += ((pred == 0) & (labels == 1)).sum().item()
                FN += ((pred == 1) & (labels == 0)).sum().item()
                total_loss += loss.item()
                total_correct += correct

        avg_val_loss = total_loss / total_samples
        val_accuracy = total_correct / total_samples

        print(
            f"Epoch {epoch}, Average Validation Loss: {avg_val_loss}, Total Validation Accuracy: {val_accuracy}\n"
        )

        # Save the model with the highest validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "trained_model_weights.pth")
            print(f"Best Model Saved\n")

        print_confusion_matrix("Validation Statistics", TP, TN, FP, FN)

        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        # Step Scheduler
        scheduler.step()

    return train_losses, train_accuracies, val_losses, val_accuracies


def test_model(model, test_loader, criterion):
    print("Testing Best Performing Epoch Weights on Training Set...")
    # Check if the specified model weights file exists
    if not os.path.exists("trained_model_weights.pth"):
        raise FileNotFoundError(
            f"Model weights file 'trained_model_weights.pth' not found."
        )

    # Load trained model weights
    print("Loading Model Weights...")
    model.load_state_dict(torch.load("trained_model_weights.pth"))
    model.eval()
    print("Model ready.\n")

    model.eval()
    total_loss, total_correct, total_samples = 0, 0, len(test_loader.dataset)
    TP, TN, FP, FN = 0, 0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            pred = logits.argmax(dim=1)
            correct = pred.eq(labels).sum().item()
            TP += ((pred == 0) & (labels == 0)).sum().item()
            TN += ((pred == 1) & (labels == 1)).sum().item()
            FP += ((pred == 0) & (labels == 1)).sum().item()
            FN += ((pred == 1) & (labels == 0)).sum().item()
            total_loss += loss.item()
            total_correct += correct

    print(f"Total Loss: {total_loss}, Total Accuracy: {total_correct/total_samples}")
    return total_loss, total_correct, TP, TN, FP, FN


def plot_performance(train_data, val_data, title, ylabel):
    plt.figure(figsize=(12, 6))
    plt.plot(train_data, label="Training", color="blue")
    plt.plot(val_data, label="Validation", color="red")
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    device = initialize_device()
    print(f"\nFETCHING DATA LOADERS\n{'='*25}\n")
    train_loader, val_loader = get_train_val_loaders()

    print(f"\nINITIALIZING MODEL\n{'='*25}\n")
    model = initialize_model(device)

    # Define Loss, Optimizer, and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=0.000001, weight_decay=0, amsgrad=True
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    print(f"\nTRAINING MODEL\n{'='*25}\n")
    train_losses, train_accuracies, val_losses, val_accuracies = train_and_validate(
        model,
        train_loader,
        val_loader,
        num_epochs=50,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    # Plotting of Training Performance
    plot_performance(train_losses, val_losses, "Cross Entropy Loss over Time", "Loss")
    plot_performance(train_accuracies, val_accuracies, "Accuracy over Time", "Accuracy")

    # Testing Phase
    print(f"\nTESTING MODEL\n{'='*25}\n")
    test_loader = get_test_loader()
    test_loss, test_acc, TP, TN, FP, FN = test_model(model, test_loader, criterion)

    print_confusion_matrix("Test Statistics", TP, TN, FP, FN)

    print("Training Completed!")
