"""
test.py

Code for testing, and loading the saved model.

Author: Atharva Gupta
Date Created: 20-10-2023
"""
import torch
import torch.nn as nn
from modules import CustomPerceiver
from datasplit import test_loader
from parameter import *

# Define the class labels
classes = ('AD', 'NC')

# Create an instance of the CustomPerceiver model with the specified parameters
net = CustomPerceiver(NUM_LATENTS, DIM_LATENTS, DEPTH_LATENT_TRANSFORMER, NUM_CROSS_ATTENDS)

# Load the pre-trained model weights
net.load_state_dict(torch.load(MODEL_PATH))

correct = 0
total = 0

# Iterate through the test data loader
with torch.no_grad():
    for i, data in enumerate(test_loader(DATA_PATH, 4), 0):
        images, labels = data
        # Calculate outputs by running images through the network
        outputs = net(images)
        # The class with the highest energy is chosen as the prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if i == 50:  # Limiting the number of iterations for illustration
            break

# Calculate and print the accuracy of the network on the test images
accuracy = 100 * correct // total
print(f'Accuracy of the network on the test images: {accuracy} %')
