import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

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
data_loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)  # Increased batch size

# Create the model
model = Net()

# Define the loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Reduced learning rate

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
    print(f'Epoch {epoch+1}, Loss: {running_loss / len(data_loader)}')  # Calculate the average loss per batch

print('Finished Training')

# Save the PyTorch model
torch.save(model.state_dict(), 'model.pth')
