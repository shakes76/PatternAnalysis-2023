import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Path to the directory containing images
DATA_PATH = './recognition/48240983_ADNI/AD_NC/train'

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Data preprocessing and loading using PyTorch
transform = transforms.Compose([transforms.Resize((150, 150)),
                                transforms.ToTensor()])

data = datasets.ImageFolder(root=DATA_PATH, transform=transform)
data_loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)

# Create the model
model = Net()

# Define the loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Lists to store loss and accuracy values
loss_values = []
accuracy_values = []

# Training the PyTorch model
for epoch in range(10):
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(data_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    average_loss = running_loss / len(data_loader)
    accuracy = correct / total
    
    print(f'Epoch {epoch+1}, Loss: {average_loss}, Accuracy: {accuracy}')
    loss_values.append(average_loss)
    accuracy_values.append(accuracy)

    # If accuracy reaches 0.8, stop training
    if accuracy >= 0.8:
        print("Desired accuracy of 0.8 reached. Stopping training.")
        break



# Plot the loss and accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(loss_values) + 1), loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(range(1, len(accuracy_values) + 1), accuracy_values)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')

plt.grid(True)
plt.tight_layout()
plt.show()
