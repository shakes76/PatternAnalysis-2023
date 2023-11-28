"""
File: predict.py
Author: Maia Josang
Description: Performs super-resolution on test data using the trained model and visualizes the results.
"""

import torch
import matplotlib.pyplot as plt
from modules import SuperResolutionModel
from dataset import ADNIDataset

# Initialize the dataset and data loaders
data_loader = ADNIDataset()
test_loader = data_loader.test_loader
test_target_loader = data_loader.test_target_loader

# Initialize the model and load trained weights
model = SuperResolutionModel()
model.load_state_dict(torch.load("super_resolution_model.pth"))
model.eval()

# Predict and visualize super-resolved images
with torch.no_grad():
    for train_batch, target_batch in zip(test_loader, test_target_loader):
        train_inputs, train_targets = train_batch 
        target_inputs, target_targets = target_batch 
        outputs = model(train_inputs)

        # Visualize input, target, and output images
        for i in range(len(train_inputs)):
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.title("Input")
            plt.imshow(train_inputs[i][0].cpu().numpy(), cmap="gray")

            plt.subplot(1, 3, 2)
            plt.title("Target (High-Resolution)")
            plt.imshow(target_inputs[i][0].cpu().numpy(), cmap="gray")

            plt.subplot(1, 3, 3)
            plt.title("Super-Resolved Output")
            plt.imshow(outputs[i][0].cpu().numpy(), cmap="gray")

            plt.show()
