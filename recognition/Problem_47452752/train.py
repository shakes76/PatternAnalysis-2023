"""
Contains the source code for training, validating, testing and saving my model.
The model should be imported from “modules.py” and the data loader should be imported from “dataset.py”.
Make sure to plot the losses and metrics during training
"""


from sympy import false
from dataset import (
    ISICDataset,
    transform,
    train_loader,
    test_loader,
    split_data,
    check_consistency,
)
from modules import UNet
from utils import dice_loss, dice_coefficient
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Subset  # for testing only TODO


# Hyper-parameters
num_epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check if the dataset is consistent
# check_consistency()
debugging = False

if debugging:
    dataset = ISICDataset(transform)
    subset_indices = list(range(200))  # debugging on first 200 samples
    subset = Subset(dataset, subset_indices)

    test_size = int(0.5 * len(subset))
    train_size = len(subset) - test_size
    train_dataset, test_dataset = split_data(subset, train_size, test_size)

    train_loader = train_loader(train_dataset, 50)
    test_loader = test_loader(test_dataset, 50)

# Loading up the dataset and applying custom augmentations
dataset = ISICDataset(transform)

# Splitting into testing and training sets
test_size = int(0.2 * len(dataset))
train_size = len(dataset) - test_size
train_dataset, test_dataset = split_data(dataset, train_size, test_size)

# Data loaders for training and testing
train_loader = train_loader(train_dataset, 100)
test_loader = test_loader(test_dataset, 100)

# Creating an instance of my UNet to be trained
model = UNet(in_channels=6, num_classes=2)
model = model.to(device)

# Setup the optimizer
optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.985)

running_loss = 0.0
print_every = 10  # Print every 10 batches.

for epoch in range(num_epochs):
    model.train()
    for i, (images, masks) in enumerate(train_loader, 1):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        masks_expanded = torch.cat((masks, 1 - masks), dim=1)
        loss = dice_loss(outputs, masks_expanded)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % print_every == 0:  # Print every `print_every` batches
            print(f"Epoch {epoch + 1}, Batch {i}: Loss = {running_loss / print_every:.4f}")
            running_loss = 0.0


# # Training loop
# for epoch in range(num_epochs):
#     model.train()  # Switch to training mode
#     for images, masks in train_loader:
#         # Move the data onto the device
#         images, masks = images.to(device), masks.to(device)

#         # Zero the parameter gradients
#         optimizer.zero_grad()

#         # Forward pass
#         outputs = model(images)
#         masks_expanded = torch.cat((masks, 1 - masks), dim=1)

#         loss = dice_loss(outputs, masks_expanded)

#         # Backward pass + optimization
#         loss.backward()
#         optimizer.step()

#     scheduler.step()  # Adjust learning rate

print("training complete")  # TODO

# Save the model
# torch.save(
#     model.state_dict(),
#     "/home/Student/s4745275/PatternAnalysis-2023/recognition/Problem_47452752/model.pth",
# )

# Switch to evaluation mode
model.eval()

# Metrics storage
dice_scores = []

# No gradient computation during evaluation
with torch.no_grad():
    for inputs, masks in test_loader:
        inputs, masks = inputs.to(device), masks.to(device)

        # Compute predictions
        outputs = model(inputs)

        # Convert outputs to binary mask
        # If sigmoid activation is used in the last layer
        predicted_masks = (outputs > 0.5).float()

        # Compute Dice coefficient
        for pred, true in zip(predicted_masks, masks):
            dice_scores.append(dice_coefficient(pred, true).item())

# Calculate average Dice score
avg_dice_score = sum(dice_scores) / len(dice_scores)

print(f"Average Dice Coefficient on Test Set: {avg_dice_score:.4f}")
