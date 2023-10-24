"""
Contains the source code for training, validating, testing and saving my model.
The model should be imported from “modules.py” and the data loader should be imported from “dataset.py”.
Make sure to plot the losses and metrics during training
"""


from dataset import ISICDataset, transform, train_loader, test_loader, split_data, check_consistency
from modules import UNet
from utils import dice_loss, dice_coefficient
import torch
import torch.optim as optim
import torch.nn.functional as F


# Hyper-parameters
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check if the dataset is consistent
check_consistency()

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

# Training loop
for epoch in range(num_epochs):
    model.train()  # Switch to training mode
    for images, masks in train_loader:
        # Move the data onto the device
        images, masks = images.to(device), masks.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        _, outputs = torch.max(model(images), 1)
        outputs = outputs.unsqueeze(1)
        # outputs = model(images)
        print(outputs.size())
        print(masks.size())

        loss = dice_loss(outputs, masks)

        # Backward pass + optimization
        loss.backward()
        optimizer.step()

    scheduler.step()  # Adjust learning rate

# Save the model
torch.save(model.state_dict(), "recognition/Problem_47452752/model.pth")

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
