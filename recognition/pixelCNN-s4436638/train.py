import os
import torch
from torch.utils.data import DataLoader
from modules import pixelCNN
from dataset import GetADNITrain

# Get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Define the path to the dataset
images_path = "/home/Student/s4436638/Datasets/AD_NC/train/*"

### Define a few training parameters
batch_size = 10
upscale_factor = 4
channels = 1
feature_size = 32
num_convs = 3
learning_rate = 1e-3

# Define our training and validation datasets
train_set = GetADNITrain(images_path, train_split=0.9, train=True)
val_set = GetADNITrain(images_path, train_split=0.9, train=False)
# Define our training and validation dataloaders
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=4, shuffle=True)
val_loader = DataLoader(dataset=val_set, batch_size=batch_size, num_workers=4, shuffle=False)

# Print out some information about the datasets
print("Num Train: " + str(train_set.arrLen))
print("Num Val: " + str(val_set.arrLen))

### Load the model
model = pixelCNN(upscale_factor, channels, feature_size, num_convs)
# Send the model to the device
model = model.to(device)

# Print the number of parameters in the model
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of trainable params: " + str(pytorch_total_params))

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


