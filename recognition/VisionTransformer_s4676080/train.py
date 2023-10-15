# This is the train file

import torch
import torch.nn as nn
import torch.optim as optim
from dataset import train_loader, test_loader
from modules import VisionTransformer
import matplotlib.pyplot as plt

# Model configurations
def initialize_model():
    model = VisionTransformer().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=2e-4)  # Introduced weight decay
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)  # Learning rate scheduler
    return model, criterion, optimizer, scheduler

def train_one_epoch(model, criterion, optimizer, train_loader):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()        
        running_loss += loss.item() * images.size(0)
    return running_loss / len(train_loader.dataset)    

# Testing loop
def test_model(model, criterion, test_loader):
    model.eval()  
    running_loss = 0.0
    correct = 0
    total = 0    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / len(test_loader.dataset), correct / total * 100

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Initialization
    model, criterion, optimizer, scheduler = initialize_model()
    
    # Training configurations
    num_epochs = 10
    train_losses = []

    print("Training started!")
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, criterion, optimizer, train_loader)
        train_losses.append(train_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")
        scheduler.step()
    print("Finished training!")
    
    print("Testing started!")
    test_loss, accuracy = test_model(model, criterion, test_loader)
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
# The accuracy is increased to 58.40% by increasing the train and test images (no hyperparameter tuning done)
