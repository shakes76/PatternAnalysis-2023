import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_data_loaders
from modules import ViT
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
    train_loader, test_loader = get_data_loaders()

    # Initialize Model
    print(f"\nINITIALIZING MODEL\n{'='*25}\n")
    print("Assigning model instance...")
    num_labels = 2  # Alzheimer's or Normal
    model = ViT(num_classes=num_labels).to(device)
    print("Model ready.")

    # Define Loss, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002, weight_decay=0.1, betas=(0.9, 0.999))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # Training and Validation Loop
    num_epochs = 10
    best_val_accuracy = 0

    print(f"\nTRAINING MODEL\n{'='*25}\n")
    for epoch in range(num_epochs):
        # Training Loop
        model.train()
        total_loss, total_correct, total_samples = 0, 0, len(train_loader.dataset)
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
            total_loss += loss.item()
            total_correct += correct

            if batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Training Loss: {loss.item()}, Training Accuracy: {correct / len(images)}"
                )

        avg_accuracy = total_correct / total_samples
        avg_total_loss = total_loss / total_samples
        print(
            f"Epoch {epoch}/{num_epochs}, Average Training Loss: {avg_total_loss}, Average Training Accuracy: {avg_accuracy}"
        )

        # Validation Loop
        model.eval()
        total_loss, total_correct, total_samples = 0, 0, len(test_loader.dataset)

        # Counters for True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN)
        TP, TN, FP, FN = 0, 0, 0, 0

        with torch.no_grad():
            for images, labels in test_loader:
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

        val_accuracy = total_correct / total_samples
        print(
            f"Epoch {epoch}, Total Validation Loss: {total_loss}, Total Validation Accuracy: {val_accuracy}\n"
        )

        # Print the counters in a table format
        print("Validation Statistics:")
        print("-----------------------")
        print(f"True Alzheimer's (TP): {TP}")
        print(f"True Normal (TN): {TN}")
        print(f"False Alzheimer's (FP): {FP}")
        print(f"False Normal (FN): {FN}")
        print("-----------------------\n")
        
        # Step the scheduler
        scheduler.step(total_loss)

        # Save the model with the highest validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "trained_model_weights.pth")
            print(f"Best Model Saved at Epoch {epoch}\n")

    print("Training Completed!")
