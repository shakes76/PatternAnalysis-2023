import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from modules import Perceiver, CrossAttention
from dataset import get_dataloaders

# Constants and hyperparameters
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.005
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PATH = './perceiver_model.pth'

# Model Initialization
model = Perceiver(
    input_dim=3 * 224 * 224,
    latent_dim=256,
    embed_dim=256,
    n_classes=2,
    num_heads=4  
).to(DEVICE)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Data Loaders
train_loader, valid_loader, test_loader = get_dataloaders("C:\\Users\\AK\\Documents\\COMP3710\\AD_NC", batch_size=BATCH_SIZE)

# Training
train_losses = []
accuracies = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        running_loss += loss.item()
    
    train_losses.append(running_loss / len(train_loader))
    accuracies.append(100 * correct_train / total_train)
    
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {accuracies[-1]:.2f}%")

# After training, evaluate on the test set
model.eval()
correct_test = 0
total_test = 0
for inputs, labels in test_loader:
    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    total_test += labels.size(0)
    correct_test += (predicted == labels).sum().item()

test_accuracy = 100 * correct_test / total_test
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(accuracies, label='Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training Accuracy over Epochs')
plt.legend()
plt.show()

# Saving model
torch.save(model.state_dict(), MODEL_PATH)
