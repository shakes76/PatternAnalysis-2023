import torch.nn as nn
import os
import numpy as np
import random
from torchvision import datasets, transforms
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from PIL import Image
import torch.optim as optim
import torch.nn.functional as F


from dataset import create_siamese_dataloader,get_classification_dataloader,get_transforms_training, get_transforms_testing
from modules import SiameseResNet, ContrastiveLoss, ClassifierNet


#----SET UP DEVICE-----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#-----LOAD TEST SET------
ROOT_DIR_TEST = "/home/groups/comp3710/ADNI/AD_NC/test"  # Modify Path if needed
test_loader = get_classification_dataloader(ROOT_DIR_TEST, batch_size=32,split_flag = False)

#----DEFINE MODEL PATHS----
# Modify paths if needed
siamese_weights = 'siamese_50.pth'
classifier_weights = "best_classifier_model_50_30_3.pth"

#---LOAD MODELS-------
siamese_model = SiameseResNet().to(device)
siamese_model.load_state_dict(torch.load(siamese_weights, map_location=device))
siamese_model.eval()

classifier = ClassifierNet(siamese_model).to(device)
classifier.load_state_dict(torch.load(classifier_weights, map_location=device))
classifier.eval()

#----DEFINE PREDICT FUNCTION-----
def predict(input_image):
    
    # Extract the embedding of the input image using Siamese Network
    embedding = siamese_model.forward_one(input_image)
    
    # Get prediction probability from classifier
    prediction_prob = classifier.classifier(embedding)

    return embedding,prediction_prob

#---EXAMPLE USAGE-------

images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

# Get a single sample image to use for prediction
sample_image_tensor = images[0].unsqueeze(0) 

# Predicting
embedding, predicted_prob_tensor = predict(sample_image_tensor)

# Extracting the probability and using round to get the class prediction 0/1
predicted_prob = predicted_prob_tensor.item()
predicted_class = round(predicted_prob)

# Extracting the true label
true_label = labels[0].item()

# Printing results
print(f"Predicted probability: {predicted_prob:.4f}")
print(f"Predicted class: {predicted_class}")
print(f"True label: {true_label}")

# Checking the accuracy of the prediction
is_correct = predicted_class == true_label
print(f"Is the prediction correct? {'Yes' if is_correct else 'No'}")



