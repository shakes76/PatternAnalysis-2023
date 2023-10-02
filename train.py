import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dataset import ADNIDataset  # Assuming your dataset class is in a file called dataset.py
from model_test import VisionTransformer  # Assuming your model class is in a file called model.py
from torch.optim.lr_scheduler import CyclicLR

import matplotlib.pyplot as plt
import random

# locking in seed
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

print ("note: trying out new lr scheduler and optimiser from  StepLR")

# Hyperparameters and configurations
learning_rate = 0.075

optimiser_choice = "SGD"
scheduler_active = True
batch_size = 16
num_epochs = 25
img_size = 256
num_workers = 2
momentum = 0.9
depth = 8  # Decreased Depth - from 12
n_heads = 16  # Modified Number of Heads
mlp_ratio = 6.0  # Modified MLP Ratio
max_patience = 7  # Stop training if the validation loss doesn't improve for 7 epochs - hyperparameter
#update 1st oct 1.55pm - changed patience to 10 from 7
test_num = 14
optim_path_dict = {"AdamW": "AdamW/", "Radam": "RAdam/", "SGD": ""}
optim_add_path = optim_path_dict[optimiser_choice]
save_model_as = "{}saved_models/best_model_{}".format(optim_add_path, test_num)
save_fig_name = "{}training_and_validation_loss_{}".format(optim_add_path, test_num)

device = "cuda" if torch.cuda.is_available() else "cpu"
print ("~~~ CONFIG ~~~")
print ("test num: ", test_num)
print ("device: ", device)
print ("batch_size: ", batch_size)
print ("num_epochs: ", num_epochs)
print ("learning_rate: ", learning_rate)
print ("model save file name: ", save_model_as)
print ("model save fig name: ", save_fig_name)
print ("optimiser: ", optimiser_choice)
print ("scheduler?: ", scheduler_active)
print ("momentum: ", momentum)
print ("max patience: ", max_patience)
print ("depth: ", depth)
print ("n_heads: ", n_heads)
print ("mlp_ratio: ", mlp_ratio)


# TRANSFORMS
# Adding more augmentations relevant to 3D brain scans such as vertical and horizontal flips
data_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
])
standard_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

# DATASET 
# Initialize the dataset
root_dir = '/home/groups/comp3710/ADNI/AD_NC'
train_dataset = ADNIDataset(root_dir=root_dir, subset='train', transform=data_transforms)

# Test dataset
test_dataset = ADNIDataset(root_dir=root_dir, subset='test', transform=standard_transforms)


# Splitting the dataset into training and validation sets based on patient IDs
unique_patient_ids = list(set('_'.join(path.split('/')[-1].split('_')[:-1]) for path, _ in train_dataset.data_paths))
validation_patient_ids = set(random.sample(unique_patient_ids, int(0.2 * len(unique_patient_ids))))  # 20% for validation
# validation_patient_ids = unique_patient_ids[0]

# example test
path_eg, label_eg = train_dataset.data_paths[0]

# the following allows one to extract the unique IDs from a path and then check if it is in the validation set
train_data_paths = [(path, label) for path, label in train_dataset.data_paths if '_'.join(path.split('/')[-1].split('_')[:-1]) not in validation_patient_ids]
val_data_paths = [(path, label) for path, label in train_dataset.data_paths if '_'.join(path.split('/')[-1].split('_')[:-1]) in validation_patient_ids]

print ("~~~ DATA PATHS CHECK ~~~")
print ("train_data_paths: {}, val_data_paths: {}".format(len(train_data_paths), len(val_data_paths)))
print ("example train_data_paths: {}, example val_data_paths: {}".format(train_data_paths[0], val_data_paths[0]))

# check there are both types of classes in teh val set
# val_set_class_types = list(set([label for _, label in val_data_paths]))
# print ("classes: {}".format(val_set_class_types))




train_dataset.data_paths = train_data_paths
val_dataset = ADNIDataset(root_dir=root_dir, subset='train', transform=data_transforms)  # Reusing the same class but with different data paths
val_dataset.data_paths = val_data_paths

# Create DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# MODEL INIT

# model = VisionTransformer()
model = VisionTransformer(
    img_size=256, 
    patch_size=16, 
    in_channels=1, 
    n_classes=1, 
    embed_dim=768,
    depth=depth,  # Increased Depth
    n_heads=n_heads,  # Modified Number of Heads
    mlp_ratio=mlp_ratio,  # Modified MLP Ratio
    qkv_bias=True, 
    p=0., 
    attn_p=0.,  # attention dropout probability
)
model = model.to(device)  # Move the model to the device (CPU or GPU)

# LOSS FUNC & OPTIM
criterion = nn.BCEWithLogitsLoss() # since doing binary classification between 2 classes (AD and NC)

if optimiser_choice == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=0.0005)
elif optimiser_choice == 'AdamW':
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0005)
else:
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=0.0005)

# Learning Rate Scheduler
if scheduler_active:
    if optimiser_choice == "AdamW":
        scheduler = CyclicLR(optimizer, base_lr=0.0001, max_lr=learning_rate, step_size_up=200, cycle_momentum=False)
    else:
        scheduler = CyclicLR(optimizer, base_lr=0.0001, max_lr=learning_rate, step_size_up=200)




# TRAINING
# Initialize lists to store the training and validation losses
train_losses = []
val_losses = []
# Initialize variables for Early Stopping
best_val_loss = float('inf')
patience = 0

# Training Loop
print ("~~~ TRAINING ~~~")
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    train_loss = 0.0
    
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        labels = labels.view(-1, 1).float()  # Reshaping from [32] to [32, 1]

        
        optimizer.zero_grad()  # Reset gradients
        
        # Forward pass
        outputs, attn_weights = model(data)
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
    
    # Update learning rate
    if scheduler_active:
        scheduler.step()

    # Validation
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():  # No need to track gradients
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            labels = labels.view(-1, 1).float()  # Reshaping from [32] to [32, 1]
            
            # Forward pass
            outputs, _ = model(data)
            
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
        torch.save(model.state_dict(), "{}.pth".format(save_model_as))
        best_val_loss = val_loss
        patience = 0  # Reset patience
    else:
        if epoch > 10:
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
plt.savefig('{}.png'.format(save_fig_name))

# TESTING
# Load the best model
model.load_state_dict(torch.load("{}.pth".format(save_model_as)))

# Testing the Final Model
print ("~~~ TESTING ~~~")
model.eval()  # Set the model to evaluation mode
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():  # No need to track gradients
    for data, labels in test_loader:
        data, labels = data.to(device), labels.to(device)
        labels = labels.view(-1, 1).float()  # Reshaping from [32] to [32, 1]

        # Forward pass
        outputs = model(data)
        
        # Apply sigmoid to get probabilities
        outputs = torch.sigmoid(outputs)
        
        # Thresholding
        predicted = (outputs > 0.5).float()

        # Compute the accuracy
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Calculate and print the test loss and accuracy
test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy}%")
