import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from dataset import get_train_val_loaders
from modules import ViT

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
if __name__ == "__main__":
    # Load Data
    print(f"\nFETCHING DATA LOADERS\n{'='*25}\n")
    train_loader, val_loader = get_train_val_loaders()

    # Initialize Model
    print(f"\nINITIALIZING MODEL\n{'='*25}\n")
    print("Assigning model instance...")
    model = ViT(
        in_channels=3,
        patch_size=14,
        emb_size=768,
        img_size=224,
        depth=14,
        n_classes=2,
    ).to(device)
    print("Model ready.")

    # Define Loss, Optimizer and Scheduler
    lr = 0.00001
    wd = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd, amsgrad=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    # Training and Validation Loop
    num_epochs = 50
    best_val_accuracy = 0

    # Used for Graphing after Training
    val_losses = []
    val_accuracies = []

    print(f"\nTRAINING MODEL\n{'='*25}\n")
    for epoch in range(num_epochs):
        # Training Loop
        model.train()
        total_loss, total_correct, total_samples = 0, 0, len(train_loader.dataset)
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

            # For Epoch Print
            total_loss += loss.item()
            total_correct += correct

            if batch_idx % 10 == 0:
                avg_batch_loss = batch_loss / (10 * len(images))
                avg_batch_accuracy = batch_correct / (10 * len(images))
                print(
                    f"Epoch {epoch}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, 10 Batch Average Training Loss: {avg_batch_loss}, 10 Batch Average Training Accuracy: {avg_batch_accuracy}"
                )
                batch_loss = 0
                batch_correct = 0

        # Validation Loop
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
                # Update counters based on predictions
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

        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        # Step the scheduler
        scheduler.step()

        # Save the model with the highest validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "trained_model_weights.pth")
            print(f"Best Model Saved at Epoch {epoch}\n")

        # ======================================
        #   Plotting of Confusion Matrix
        # ======================================

        # Print the counters in a table format per Epoch Val
        print("Validation Statistics:")
        print("-----------------------")
        print(f"|{'True AD (TP)':<25}|{TP:>5}|")
        print(f"|{'True NC (TN)':<25}|{TN:>5}|")
        print(f"|{'False AD (FP)':<25}|{FP:>5}|")
        print(f"|{'False NC (FN)':<25}|{FN:>5}|")
        print("-----------------------\n")

    # ======================================
    #   Plotting of Training Performance
    # ======================================

    # Plotting Validation Loss
    plt.figure(figsize=(12, 6))
    plt.plot(val_losses, label="Validation Loss", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Validation Loss over Time")
    plt.legend()
    plt.show()

    # Plotting Validation Accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(val_accuracies, label="Validation Accuracy", color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy over Time")
    plt.legend()
    plt.show()
    print("Training Completed!")
