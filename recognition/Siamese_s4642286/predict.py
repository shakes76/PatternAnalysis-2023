# Name: predict.py
# Student: Ethan Pinto (s4642286)
# Description: Shows example usage of trained model.

import torch
from modules import SiameseNetwork

# Import the trained model
siamese = SiameseNetwork()


# step 2: load new version of siamese, then load saved weights in eval mode. This will make up the classifier.
siamese.load_state_dict(torch.load("./model"))
siamese.eval()

# Move the model to the desired device (e.g., GPU)
# loaded_siamese_net.to("cuda")


# Create the test dataloader in here and use it to get a batch of images.


# Use the model to predict the class of the images in the batch.


# Need a minimum accuracy of 0.8 on the test set.


# Step 3: classification - take in individual images into snn, then write an mlp which takes in a feature vector of 128 and then classifies into one of two classes.
