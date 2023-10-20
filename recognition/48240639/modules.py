"""
Created on Wednesday October 18
Siamese Network using PyTorch

This code defines a Siamese Network architecture for image similarity comparison.
The network is built upon the ResNet-18 architecture with modifications to handle
grayscale images. It includes custom weight initialization and forward pass methods.

@author: Aniket Gupta 
@ID: s4824063
"""

import torch
import torch.nn as nn
import torchvision


class SiameseNN(nn.Module):
    def __init__(self):
        """
        Initialize a Siamese Network model based on ResNet-18 architecture.
        The network is designed for image similarity comparison.

        The ResNet-18 architecture is adapted to handle grayscale images, such as those in the ADNI dataset.

        The model includes custom weight initialization for linear layers and defines the forward pass methods.

        """
        super(SiameseNN, self).__init__()

        # ResNet18 model
        self.resnet = torchvision.models.resnet18(weights=None)

        # Modify the first convolutional layer to accommodate grayscale images
        # As ResNet-18 expects (3,x,x) input, whereas ADNI images are grayscale (1,x,x)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc_in_features = self.resnet.fc.in_features

        # Discard the last layer of ResNet-18 (the linear layer before avgpool)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        # Introduce linear layers for comparing features from two input images
        self.fc = nn.Sequential(    # Create a sequential neural network for feature comparison
            nn.Linear(self.fc_in_features * 2, 256),  # Linear layer to process concatenated features from two input images
            nn.ReLU(inplace=True),  # Apply ReLU activation function for non-linearity
            nn.Linear(256, 1),# Linear layer to produce a single output for similarity comparison
        )

        self.sigmoid = nn.Sigmoid()

        # Initialize the model's weights
        self.resnet.apply(self.initialize_weight)
        self.fc.apply(self.initialize_weight)

    def forward_one(self, x):
        """
        Forward pass for a single image to obtain its features.

        Args:
        - x: torch.Tensor
            Input image to extract features from.

        Returns:
        - output: torch.Tensor
            Extracted features from the input image.
        """
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def initialize_weight(self, m):
        """
        Initialize the weights of the linear layers.

        Args:
        - m: torch.nn.Module
            The module for which weights are initialized.

        Returns:
        None
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, input1, input2):
        """
        Forward pass for comparing two input images and determining their similarity.

        Args:
        - input1: torch.Tensor
            First input image.
        - input2: torch.Tensor
            Second input image.

        Returns:
        - output: torch.Tensor
            Similarity score between the two input images.
        """
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)

        output = torch.cat((output1, output2), 1)
        output = self.fc(output)
        output = self.sigmoid(output)

        return output
