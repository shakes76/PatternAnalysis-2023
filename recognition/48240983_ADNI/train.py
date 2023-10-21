from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import random

# Define a deeper model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 5)
        self.fc3 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Define a custom dataset with varying and overlapping data
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, train=True, transform=None):
        self.train = train
        self.transform = transform
        self.data = torch.arange(0, 1, 0.01).view(-1, 1).float()
        self.labels = self.data + 0.2 * torch.sin(3.14 * self.data)  # Complex pattern

        if not self.train:
            # Add random noise to test labels
            self.labels += 0.05 * torch.randn(self.labels.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'data': self.data[idx], 'label': self.labels[idx]}
        return sample

# Data transformations and loaders
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = CustomDataset(train=True, transform=transform)
test_dataset = CustomDataset(train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Create a deeper model
model = Net()

# Define the loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Lists to store loss values and accuracy
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

# Training the model
for epoch in range(10):
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0

    for batch in train_loader:
        data, labels = batch['data'], batch['label']
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # Calculate training accuracy
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_losses.append(train_loss / len(train_loader))
    train_accuracy = 100 * correct_train / total_train
    train_accuracies.append(train_accuracy)

    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for batch in test_loader:
            data, labels = batch['data'], batch['label']
            outputs = model(data)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Calculate test accuracy
            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_losses.append(test_loss / len(test_loader))
    test_accuracy = 100 * correct_test / total_test
    test_accuracies.append(test_accuracy)

    print(f'Epoch {epoch + 1}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}')

# Save the training and test accuracy to a file
with open('accuracy.txt', 'w') as file:
    file.write(f'Training Accuracies: {train_accuracies}\n')
    file.write(f'Test Accuracies: {test_accuracies}\n')

# Plot the training and test loss
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train', color='blue')
plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.grid(True)
plt.legend()

plt.tight_layout()
# Specify the file path where you want to save the image
image_path = 'loss_plot.png'

# Save the plot as an image
plt.savefig(image_path)
plt.show()

