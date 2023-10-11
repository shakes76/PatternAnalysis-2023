import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from dataset import ISICDataset  # Assuming dataset.py is in the same directory
from modules import MaskRCNN  # Assuming modules.py is in the same directory
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Initialize the dataset
train_dataset = ISICDataset(path="E:/comp3710/ISIC2018", type="Training")
val_dataset = ISICDataset(path="E:/comp3710/ISIC2018", type="Validation")

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Initialize the model
num_classes = 2  # For binary classification, we have 2 classes: 0 and 1
model = MaskRCNN(num_classes)

# If CUDA is available, move the model to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model.to(device)

# Define the loss functions
classification_loss = nn.CrossEntropyLoss()
bbox_loss = nn.SmoothL1Loss()
mask_loss = nn.BCEWithLogitsLoss()

# Initialize the optimizer
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
for epoch in range(10):  # Number of epochs
    for i, (images, masks) in enumerate(train_loader):

        images, masks = images.to(device), masks.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        classification, boxes, predicted_masks = model(images)

        # Calculate the loss
        loss_cls = classification_loss(classification)
        loss_bbox = bbox_loss(boxes)
        loss_mask = mask_loss(predicted_masks, masks)

        # Combine the losses
        total_loss = loss_cls + loss_bbox + loss_mask

        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/10], Step [{i+1}/{len(train_loader)}], Loss: {total_loss.item():.4f}")

# Save the model
torch.save(model.state_dict(), "mask_rcnn_model.pth")
