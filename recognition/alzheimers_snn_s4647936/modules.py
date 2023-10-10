import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        
        # Define convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)  # assuming grayscale images
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        
        # Define a placeholder for fully connected layers (we'll adjust these later based on input image size)
        self.fc1 = nn.Linear(64*50*50, 256)  # 50x50 is a placeholder; adjust based on actual output size
        self.fc2 = nn.Linear(256, 2)  # 2-dimensional output for simplicity
        
    def forward(self, x):
        # Apply the convolutional layers with ReLU and max pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # kernel size 2
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        
        # Apply the fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

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

