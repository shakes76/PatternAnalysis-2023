"""
predict.py: Image classification using a Siamese Network and a Classifier.

Author: Rachit Chaurasia (s4823870)
Date: 20/10/2023

This script loads a pre-trained Siamese Network model and uses it to perform image classification.
The classifier is trained to classify images based on Siamese network embeddings.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from modules import SiameseNetwork
from dataset import load_classify_data
from sklearn.metrics import accuracy_score

# Load the pre-trained Siamese Network
snn_path = 'SNN.pth'
siamese_network = SiameseNetwork()
siamese_network.load_state_dict(torch.load(snn_path,map_location=torch.device('cpu')))
siamese_network.eval()

# Define a classifier that takes Siamese network embeddings and performs classification
class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        """
        Initialize a classifier for image classification.

        Args:
            input_size (int): The input feature size (embedding size).
            hidden_size (int): The size of the hidden layer.
            num_classes (int): The number of classes for classification.
        """
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        Forward pass through the classifier.

        Args:
            x (Tensor): Input tensor with Siamese network embeddings.

        Returns:
            Tensor: Output tensor representing class scores.
        """
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Hyperparameters
input_size = 2  
hidden_size = 128
num_classes = 2  # 2 classes: Normal and AD
batch_size = 32

# Load the data for classification
classify_data_loader = load_classify_data(testing=True, batch_size=batch_size)

# Initialize and train the classifier
classifier = Classifier(input_size, hidden_size, num_classes)

# Check if CUDA (GPU) is available and move the classifier to the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)

# Training loop for the classifier
num_epochs = 45  # This can be adjusted
for epoch in range(num_epochs):
    classifier.train()
    for images, _, labels in classify_data_loader:
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.long().to(device)
        
        # Forward pass using Siamese Network
        with torch.no_grad():
            embeddings = siamese_network.forward_one(images)

        # Forward pass through the classifier
        outputs = classifier(embeddings)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluate the classifier on the test set
classifier.eval()
test_loss = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, _, labels in classify_data_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass using Siamese Network
        embeddings = siamese_network.forward_one(images)

        # Forward pass through the classifier
        outputs = classifier(embeddings)

        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f'Test Accuracy: {accuracy:.2f}')
