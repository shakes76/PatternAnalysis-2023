import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

"""
Feature Extractor
- Sub-network responsible for extracting features from an input image.
- Implemented with simple convolutional neural network (CNN) structure.
"""
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        
        # Define convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)  # assuming grayscale images
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5)
        
        # Define fully connected layers (adjust based on input image size)
        self.fc1 = nn.Linear(28*26*128, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 2)
        
    def forward(self, x):
        # Apply the convolutional layers with ReLU and max pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # kernel size 2
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        
        # Flatten the tensor
        x = x.view(x.size(0), 28*26*128)
        
        # Apply the fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

"""
Siamese Network
- Uses two copies of 'FeatureExtractor' to process two images
"""
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        # Use the same feature extractor for both inputs
        self.feature_extractor = FeatureExtractor()
        
    def forward_one(self, x):
        # Forward pass for one input
        return self.feature_extractor(x)
        
    def forward(self, input1, input2):
        # Forward pass for both inputs
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2
    
"""
Determine dataset image dimensions
"""
# Path to images
image_path = "/home/Student/s4647936/PatternAnalysis-2023/recognition/alzheimers_snn_s4647936/AD_NC/train/AD/336537_97.jpeg"  # Adjust this path as per your directory structure

# Open the image and determine its size
image = Image.open(image_path)
width, height = image.size

# print(f"Image dimensions: {width} x {height}") # Result is 256 x 240

"""
Triplet Loss implementation
- Beneficial to choose "hard" triplets
- The anchor and positive would be two different "slices" from the same patient
- The negative would be a "slice" from a different patient
"""
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)  # Euclidean distance
        distance_negative = (anchor - negative).pow(2).sum(1)  # Euclidean distance
        losses = nn.functional.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()
    
class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(2, 64)  # Embedding size is 2
        self.fc2 = nn.Linear(64, 2)  # Output is 2 (AD or NC)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
