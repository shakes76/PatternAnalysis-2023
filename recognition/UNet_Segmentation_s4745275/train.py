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
import numpy as np
from modules import UNet
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from utils import dice_loss, dice_coefficient
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Subset  # for debugging only TODO


# Hyper-parameters
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Conditional parameters used for debugging
check = False
debugging = False
saving = True
validating = True


def evaluate_model(model, data_loader, device):
    # set the model to evaluation mode
    model.eval()

    # List to store individual dice scores for each sample
    all_dice_scores = []

    # No gradient computation during evaluation
    with torch.no_grad():
        for images, masks in data_loader:
            images, masks = images.to(device), masks.to(device)

            # Compute predictions
            outputs = model(images)

            # Convert predictions to binary using threshold
            outputs = (outputs > 0.5).float()

            # Compute and store the dice coefficients
            batch_dice_scores = dice_coefficient(outputs, masks)
            all_dice_scores.extend(batch_dice_scores.cpu().numpy())

    avg_dice_score = np.mean(all_dice_scores)
    min_dice_score = np.min(all_dice_scores)

    return avg_dice_score, min_dice_score


# Check if the dataset is consistent
if check:
    check_consistency()

# Construct debugging datasets (small subset for fast tests)
if debugging:
    num_epochs = 2
    subset_size = 500

    dataset = ISICDataset(transform)
    subset_indices = list(range(subset_size))
    subset = Subset(dataset, subset_indices)

    train_size = int(subset_size * 0.6)
    val_size = int(subset_size * 0.2)
    test_size = subset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        subset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, 32, True)
    validation_loader = DataLoader(val_dataset, 100, False)
    test_loader = DataLoader(test_dataset, 100, False)

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

# Variables used for training feedback and validation:
running_loss = 0.0
no_improvement = 0
print_every = 1  # Print every 10 batches.
best_val_similarity = 0.0
best_val_minimum = 0.0
patience = 8  # Number of epochs to wait before stopping

for epoch in range(num_epochs):
    # Set the model to training mode
    model.train()

    for i, (images, masks) in enumerate(train_loader, 1):
        # Move the data onto the device
        images, masks = images.to(device), masks.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        # print(f"outputs = {outputs.size()}")
        # print(f"masks = {masks.size()}")

        loss = dice_loss(outputs, masks)

        # Backward pass + optimization
        loss.backward()
        optimizer.step()

        # Keep track of the running loss for testing feedback
        running_loss += loss.item()
        if i % print_every == 0:  # Print every `print_every` batches
            print(
                f"Epoch {epoch + 1}, Batch {i}: Loss = {running_loss / print_every:.4f}"
            )
            running_loss = 0.0

    if validating:
        # Evaluate the model using the validation set
        dice_similarity, dice_minimum = evaluate_model(model, validation_loader, device)
        val_loss = 1 - dice_similarity
        # Print out the validation metrics
        print(
            f"Validation metrics during epoch {epoch + 1}, loss = {val_loss:.4f}, similarity = {dice_similarity:.4f}, minimum = {dice_minimum:.4f}"
        )

        # Model checkpointing
        if dice_similarity > best_val_similarity and dice_minimum >= best_val_minimum:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            no_improvement = 0
        else:
            no_improvement += 1

    # Early stoppage if the model hasn't improved in `patience` epochs
    if no_improvement > patience:
        break

    scheduler.step()  # Adjust learning rate


print("training complete")

# Save the model
if saving:
    torch.save(
        model.state_dict(),
        "/home/Student/s4745275/PatternAnalysis-2023/recognition/Problem_47452752/model.pth",
    )

avg_dice_score, min_dice_score = evaluate_model(model, test_loader, device)

print(f"Average Dice Coefficient on Test Set: {avg_dice_score:.4f}")
print(f"Minimum Dice Coefficient on Test Set: {min_dice_score:.4f}")
