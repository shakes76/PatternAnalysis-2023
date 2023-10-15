# This is the train file

import torch
import torch.nn as nn
import torch.optim as optim
from dataset import train_loader, test_loader
from modules import VisionTransformer
import matplotlib.pyplot as plt

# Checking for GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model configurations
model = VisionTransformer().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=2e-4)  # Introduced weight decay
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)  # Learning rate scheduler

# Training loop
num_epochs = 10
train_losses = []
test_losses = []

print("training started!")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()        
        outputs = model(images)
        loss = criterion(outputs, labels)
        #print(loss)
        loss.backward()
        optimizer.step()        
        running_loss += loss.item() * images.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)    
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")    
    scheduler.step()
print("finished training!")

# Testing loop
model.eval()
test_losses = []
running_loss = 0.0
correct = 0
total = 0

print("Testing started!")
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_loss = running_loss / len(test_loader.dataset)
test_losses.append(test_loss)
accuracy = correct / total * 100
print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")

# The accuracy is increased to 58.40% by increasing the train and test images (no hyperparameter tuning done)
