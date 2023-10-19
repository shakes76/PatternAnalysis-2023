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

# Lists to store loss values
loss_values = []

# Training the PyTorch model
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(data_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    average_loss = running_loss / len(data_loader)
    print(f'Epoch {epoch+1}, Loss: {average_loss}')
    loss_values.append(average_loss)

print('Finished Training')

# Plot the loss
plt.plot(range(1, 11), loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)
plt.show()
