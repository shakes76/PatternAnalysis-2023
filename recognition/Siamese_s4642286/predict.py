# Name: predict.py
# Student: Ethan Pinto (s4642286)
# Description: Shows example usage of trained model.

import torch
from modules import SiameseNetwork

# Import the trained model
loaded_siamese_net = SiameseNetwork()

# Load the trained weights into the model
loaded_siamese_net.load_state_dict(torch.load("./model"))

# Move the model to the desired device (e.g., GPU)
loaded_siamese_net.to("cuda")


# Create the test dataloader in here and use it to get a batch of images.


# Use the model to predict the class of the images in the batch.


# Need a minimum accuracy of 0.8 on the test set.