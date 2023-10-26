"""
Contains the source code for training, validating, testing and saving my model.
The model should be imported from “modules.py” and the data loader should be imported from “dataset.py”.
Make sure to plot the losses and metrics during training
"""


from dataset import (
    ISICDataset,
    transform,
    check_consistency,
)
from modules import UNet
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from utils import dice_loss, dice_coefficient
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Subset  # for debugging only TODO


# Hyper-parameters
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Conditional parameters used for debugging
check = False
debugging = True
saving = False
validating = False

# Check if the dataset is consistent
if check:
    check_consistency()

# Construct smaller debugging datasets
if debugging:
    num_epochs = 1
    dataset = ISICDataset(transform)
    subset_indices = list(range(40))  # debugging on first 40 samples
    subset = Subset(dataset, subset_indices)

    test_size = int(0.5 * len(subset))
    train_size = len(subset) - test_size
    train_dataset, test_dataset = random_split(subset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, 10)
    test_loader = DataLoader(test_dataset, 10)

# Construct full datasets
if not debugging:
    # Loading up the dataset and applying custom augmentations
    dataset = ISICDataset(transform)

    total_size = len(dataset)

    # Splitting into training, validation and testing sets
    train_size = int(total_size * 0.7)
    val_size = int(total_size * 0.2)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Data loaders for training, validation and testing
    train_loader = DataLoader(train_dataset, 32, True)
    validation_loader = DataLoader(val_dataset, 100, False)
    test_loader = DataLoader(test_dataset, 100, False)

# Creating an instance of my UNet to be trained
model = UNet(in_channels=6, num_classes=1)
model = model.to(device)

# Setup the optimizer
optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.985)

print("Training loop:")

# Variables used for training feedback:
if validating:
    running_loss = 0.0
    print_every = 10  # Print every 10 batches.
    best_val_loss = float('inf')
    patience = 3 # Number of epochs to wait before stopping

for epoch in range(num_epochs):
    model.train()
    for i, (images, masks) in enumerate(train_loader, 1):
        # Move the data onto the device
        images, masks = images.to(device), masks.to(device)
        print(f"masks = {masks.size()}")
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        print(f"outputs = {outputs.size()}")
        
        binary_output = (outputs > 0.5).float()

        loss = dice_loss(binary_output, masks)

        # Backward pass + optimization
        loss.backward()
        optimizer.step()

    if validating:

        running_loss += loss.item()
        if i % print_every == 0:  # Print every `print_every` batches
            print(
                f"Epoch {epoch + 1}, Batch {i}: Loss = {running_loss / print_every:.4f}"
            )
            running_loss = 0.0

        # Evaluate the model using the validation set
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for images, masks in validation_loader:
                # Load the validation data onto the device
                images, masks = images.to(device), masks.to(device)
                
                # Evaluate the loss
                outputs = model(images)
                masks_expanded = torch.cat((masks, 1 - masks), dim=1)
                loss = dice_loss(outputs, masks_expanded)
                val_loss += loss.item()

        val_loss /= len(validation_loader)
        print(f"Epoch {epoch + 1}, Validation Loss = {val_loss:.4f}")

        # Model checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            no_improvement = 0
        else:
            no_improvement += 1

    scheduler.step()  # Adjust learning rate


print("training complete")

# Save the model
if saving:
    torch.save(
        model.state_dict(),
        "/home/Student/s4745275/PatternAnalysis-2023/recognition/Problem_47452752/model.pth",
    )

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
        predicted_masks = (outputs > 0.5).float()

        # Compute Dice coefficient
        for pred, true in zip(predicted_masks, masks):
            dice_scores.append(dice_coefficient(pred, true).item())

# Calculate average Dice score
avg_dice_score = sum(dice_scores) / len(dice_scores)

print(f"Average Dice Coefficient on Test Set: {avg_dice_score:.4f}")
