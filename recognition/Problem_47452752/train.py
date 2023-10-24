# containing the source code for training, validating, testing and saving your model.
# The model should be imported from “modules.py” and the data loader should be imported from “dataset.py”.
# Make sure to plot the losses and metrics during training

from dataset import ISICDataset, transform, train_loader, test_loader, split_data
from modules import UNet
from utils import dice_loss
import torch
import torch.optim as optim
import torch.nn.functional as F


# Hyper-parameters
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading up the dataset and applying custom augmentations
dataset = ISICDataset(transform)

# Splitting into testing and training sets
test_size = int(0.2 * len(dataset))
train_size = len(dataset) - test_size
train_dataset, test_dataset = split_data(dataset, train_size, test_size)

#
train_loader = train_loader(train_dataset, 100)
test_loader = test_loader(test_dataset, 100)

# Creating an instance of my UNet to be trained
model = UNet(in_channels=6, num_classes=2)
model = model.to(device)

# Setup the optimizer
optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.985)

for epoch in range(num_epochs):
    model.train()  # Switch to training mode
    for images, masks in train_loader:
        # Move the data onto the device
        images, masks = images.to(device), masks.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = dice_loss(outputs, masks)

        # Backward pass + optimization
        loss.backward()
        optimizer.step()

    scheduler.step()  # Adjust learning rate
