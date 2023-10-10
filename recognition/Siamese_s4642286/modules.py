"""
Name: modules.py
Student: Ethan Pinto (s4642286)
Description: Contains the source code of the components of the model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print('CUDA not available. Running on CPU...')

print(device)


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # CNN and Pooling layers
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=10), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=7), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=4), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=4), nn.ReLU()
        )

        self.embedding_layer = nn.Sequential(
            nn.BatchNorm1d(128 * 1 * 1),
            nn.Flatten(),
            nn.Linear(128 * 1 * 1, 50),
            nn.ReLU()
        )

        # Classify whether the images belong to the same class or different classes
        self.fc = nn.Sequential(
            nn.BatchNorm1d(50),
            nn.Linear(50, 1),
            nn.Sigmoid()
        )

    def forward_one(self, x):
        # Forward pass through one branch of the Siamese network
        x = self.cnn(x)
        x = self.embedding_layer(x)
        return x

    def forward(self, input1, input2):
        # Forward pass through both branches of the Siamese network
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)

        # Compute L1 Distance between two embeddings
        distance = torch.abs(output1 - output2)

        # Classify whether the images belong to the same class or different classes
        prediction = self.fc(distance)
        return prediction



