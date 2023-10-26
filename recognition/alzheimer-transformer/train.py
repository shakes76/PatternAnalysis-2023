'''
contains the source code for training, validating, testing and saving the model.
The model is be imported from “modules.py” and the data loader is imported from “dataset.py”. 
Make sure to plot the losses and metrics during training.
'''

from dataset import get_alzheimer_dataloader
from modules import ViT
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import argparse

class_names = ["AD", "NC"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, dataloader, criterion, optimizer, device):
    """One epoch's worth of training."""
    model.train()
    total_loss = 0.0
    total_correct = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total_correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = total_correct / len(dataloader.dataset)

    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    """Evaluate the model without updating weights."""
    model.eval()
    total_loss = 0.0
    total_correct = 0

    # no need to track gradients here
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = total_correct / len(dataloader.dataset)

    return avg_loss, accuracy

def main():

    parser = argparse.ArgumentParser()
    
    # Add arguments for hyperparameters
    parser.add_argument('--lr', type=float, default=0.00005, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.03, help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--model_size', type=str, default='base', help='base, large, huge')
    parser.add_argument('--patch_size', type=int, default=16, help='Number of pixels in a patch')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout in ViT')
    parser.add_argument('--img_size', type=int, default=224, help='Size to resize image to')
    parser.add_argument('--path', type=str, default="./dataset/AD_NC", help='Path to the dataset')
    

    # Parse the arguments
    args = parser.parse_args()

    # Now use the arguments
    lr = args.lr
    weight_decay = args.weight_decay
    num_epochs = args.num_epochs
    model_size = args.model_size
    patch_size = args.patch_size
    batch_size = args.batch_size
    dropout = args.dropout
    img_size = args.img_size
    path = args.path

    train_loader, val_loader, test_loader = get_alzheimer_dataloader(batch_size=batch_size, img_size=img_size, path=path)

    model = ViT(model_type=model_size, patch_size=patch_size, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)

    # initialize best validation accuracy at 0
    best_val_acc = 0.0

    # loop through the number of epochs
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train: Loss {train_loss:.4f}, Accuracy {train_acc:.4f}")
        print(f"Validation: Loss {val_loss:.4f}, Accuracy {val_acc:.4f}")

        # save model if it performs better on the validation set
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'vit-best_val_acc{best_val_acc*100:.2f}%.pth')
            print(f"New best validation accuracy ({best_val_acc*100:.2f}%) achieved! Model saved.")

    # after all epochs, test the model performance on test data
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test: Loss {test_loss:.4f}, Accuracy {test_acc:.4f}")

if __name__ == "__main__":
    main()