"""
train.py: Initializes, trains and test the VisionTransformer on AD 
and NC image data. After training, the model is saved as well. It also provides visualization of training 
and testing metrics.
"""

# importing libraries
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import train_loader, test_loader
from modules import VisionTransformer
import matplotlib.pyplot as plt

# Model configurations
def initialize_model():
    """
    Initialize the model, criterion, optimizer, and learning rate scheduler.

    Returns:
    - model: An instance of VisionTransformer.
    - criterion: Cross-entropy loss for classification.
    - optimizer: Adam optimizer.
    - scheduler: Learning rate scheduler.
    """
    model = VisionTransformer().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=2e-4)  
    
    # Implement learning rate scheduler for dynamic adjustment of learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.7, verbose=True)
    torch.cuda.empty_cache()
    return model, criterion, optimizer, scheduler

def train_one_epoch(model, criterion, optimizer, train_loader):
    """
    Train the model for one epoch.

    Parameters:
    - model: Model to be trained.
    - criterion: Loss function.
    - optimizer: Optimization algorithm.
    - train_loader: DataLoader for the training set.

    Returns:
    - running_loss: Total loss for this epoch.
    - correct_train: Number of correctly classified instances.
    - total_train: Total instances in the dataset.
    """
    # used chatgpt to get help in writing the training loop (basically for gradient clipping)
    model.train()
    running_loss, correct_train, total_train = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()    
        outputs = model(images)
        _, predicted_train = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping to avoid large gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    
    return running_loss, correct_train, total_train    

# Testing loop
def test_model(model, criterion, test_loader):
    """
    Evaluate the model's performance on the test set.

    Parameters:
    - model: Model to be tested.
    - criterion: Loss function.
    - test_loader: DataLoader for the test set.

    Returns:
    - running_loss: Total loss for the test set.
    - correct_test: Number of correctly classified instances.
    - total_test: Total instances in the test dataset.
    """
    model.eval()
    running_loss, correct_test, total_test = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted_test = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted_test == labels).sum().item()
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)    
    return running_loss, correct_test, total_test

def plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies):
    """
    Visualize training and testing metrics across epochs and save the fig.

    Parameters:
    - train_losses: List of training losses.
    - test_losses: List of test losses.
    - train_accuracies: List of training accuracies.
    - test_accuracies: List of test accuracies.
    """
    epochs = list(range(1, len(train_losses) + 1))  
    
    plt.figure(figsize=(10, 4))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")  
    plt.plot(epochs, test_losses, label="Test Loss")  
    plt.legend()
    plt.title("Losses over Epochs")
    plt.xticks(epochs)  
    plt.savefig('/content/drive/MyDrive/losses_plot.png')
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Train Accuracy")  
    plt.plot(epochs, test_accuracies, label="Test Accuracy")  
    plt.legend()
    plt.title("Accuracies over Epochs")
    plt.xticks(epochs)  
    plt.savefig('/content/drive/MyDrive/accuracies_plot.png')
    plt.show()

# main function
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Initialization
    model, criterion, optimizer, scheduler = initialize_model()
    
    # Training loop configurations
    num_epochs = 5
    train_losses, train_accuracies, test_losses, test_accuracies = [], [], [], []
    patience = 5
    best_test_loss = float('inf')
    counter = 0    
    #print("Training started!")
    for epoch in range(1, num_epochs + 1):  # Changed the range to fix the numbering issue
        train_loss, correct_train, total_train = train_one_epoch(model, criterion, optimizer, train_loader)
        test_loss, correct_test, total_test = test_model(model, criterion, test_loader)

        train_losses.append(train_loss / len(train_loader.dataset))
        train_accuracies.append(100 * correct_train / total_train)
        test_losses.append(test_loss / len(test_loader.dataset))
        test_accuracies.append(100 * correct_test / total_test)        
        print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.2f}%, Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_accuracies[-1]:.2f}%")
        
        scheduler.step(test_losses[-1])

        # Early stopping
        if test_losses[-1] < best_test_loss:
            best_test_loss = test_losses[-1]
            counter = 0
            torch.save(model.state_dict(), '/content/drive/MyDrive/best_model.pth')
            print("Model saved!")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered!")
                break

    #print("Training finished!")
    plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies)
    print(f"Best Training Accuracy: {max(train_accuracies):.2f}%")
    print(f"Best Test Accuracy: {max(test_accuracies):.2f}%")
