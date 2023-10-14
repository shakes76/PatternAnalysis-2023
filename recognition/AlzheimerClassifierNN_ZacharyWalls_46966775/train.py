import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_data_loaders
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
    train_loader, test_loader = get_data_loaders()

    # Initialize Model
    print(f"\nINITIALIZING MODEL\n{'='*25}\n")
    num_labels = 2  # Alzheimer's or Normal
    model = ViT(num_classes=num_labels).to(device)
    print("Model ready.")

    # Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)

    # Training and Validation Loop
    num_epochs = 10
    best_val_accuracy = 0

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

        train_accuracy = total_correct / total_samples
        print(
            f"Epoch {epoch}/{num_epochs}, Total Training Loss: {total_loss}, Total Training Accuracy: {train_accuracy}"
        )

        # Validation Loop
        model.eval()
        total_loss, total_correct, total_samples = 0, 0, len(test_loader.dataset)
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                pred = logits.argmax(dim=1)
                correct = pred.eq(labels).sum().item()
                total_loss += loss.item()
                total_correct += correct

        val_accuracy = total_correct / total_samples
        print(
            f"Epoch {epoch}/{num_epochs}, Total Validation Loss: {total_loss}, Total Validation Accuracy: {val_accuracy}"
        )

        # Save the model with the highest validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "trained_model_weights.pth")
            print(f"Best Model Saved at Epoch {epoch}")

    print("Training Completed!")
