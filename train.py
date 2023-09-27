import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dataset import ADNIDataset  # Assuming your dataset class is in a file called dataset.py
from model_test import VisionTransformer  # Assuming your model class is in a file called model.py
import matplotlib.pyplot as plt



# Hyperparameters and configurations
learning_rate = 0.001
batch_size = 32
num_epochs = 50
num_workers = 2
device = "cuda" if torch.cuda.is_available() else "cpu"
print ("device: ", device)

# TRANSFORMS
# Adding more augmentations relevant to 3D brain scans such as vertical and horizontal flips
data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
])

# DATASET 
# Initialize the dataset
root_dir = 'home/groups/comp3710/ADNI/AD_NC'
train_dataset = ADNIDataset(root_dir=root_dir, subset='train', transform=data_transforms)

# Test dataset
test_dataset = ADNIDataset(root_dir=root_dir, subset='test', transform=data_transforms)


# Splitting the dataset into training and validation sets based on patient IDs
unique_patient_ids = list(set('_'.join(fname.split('_')[:-1]) for fname, _ in train_dataset.data_paths))
validation_patient_ids = set(random.sample(unique_patient_ids, int(0.2 * len(unique_patient_ids))))  # 20% for validation

train_data_paths = [(path, label) for path, label in train_dataset.data_paths if '_'.join(path.split('/')[-1].split('_')[:-1]) not in validation_patient_ids]
val_data_paths = [(path, label) for path, label in train_dataset.data_paths if '_'.join(path.split('/')[-1].split('_')[:-1]) in validation_patient_ids]

train_dataset.data_paths = train_data_paths
val_dataset = ADNIDataset(root_dir=root_dir, subset='train', transform=data_transforms)  # Reusing the same class but with different data paths
val_dataset.data_paths = val_data_paths

# Create DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# MODEL INIT
model = VisionTransformer()
model = model.to(device)  # Move the model to the device (CPU or GPU)

# LOSS FUNC & OPTIM
criterion = nn.BCEWithLogitsLoss() # since doing binary classification between 2 classes (AD and NC)
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # may have to try SGD eventually

# Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)


# TRAINING
# Initialize lists to store the training and validation losses
train_losses = []
val_losses = []
# Initialize variables for Early Stopping
best_val_loss = float('inf')
patience = 0
max_patience = 7  # Stop training if the validation loss doesn't improve for 7 epochs - hyperparameter

# Training Loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    train_loss = 0.0
    
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        
        optimizer.zero_grad()  # Reset gradients
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
    
    # Update learning rate
    scheduler.step()

    # Validation
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():  # No need to track gradients
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(data)
            
            # Compute the loss and accuracy
            loss = criterion(outputs, labels)
            val_loss += loss.item()


    # Append losses for plotting
    train_losses.append(train_loss/len(train_loader))
    val_losses.append(val_loss/len(val_loader))

    # Logging
    print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}")

    # Save the model if it has the best validation loss so far
    if val_loss < best_val_loss:
        print(f"Validation loss improved from {best_val_loss} to {val_loss}. Saving model.")
        torch.save(model.state_dict(), "best_model.pth")
        best_val_loss = val_loss
        patience = 0  # Reset patience
    else:
        print("Validation loss did not improve. Patience:", patience)
        patience += 1

    # Early stopping
    if patience >= max_patience:
        print("Stopping early due to lack of improvement in validation loss.")
        break

# Plotting the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Losses')

# Save the plot
plt.savefig('training_and_validation_loss.png')

# TESTING
# Load the best model
model.load_state_dict(torch.load("best_model.pth"))

# Testing the Final Model
model.eval()  # Set the model to evaluation mode
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():  # No need to track gradients
    for data, labels in test_loader:
        data, labels = data.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(data)
        
        # Compute the loss and accuracy
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Calculate and print the test loss and accuracy
test_loss /= len(test_loader)
test_accuracy = 100 * correct / total
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}%")
