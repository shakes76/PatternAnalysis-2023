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
import torchvision.models as models

class SiameseNN(nn.Module):
    def __init__(self):
        super(SiameseNN, self).__init()
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.fc_in_features = self.backbone.fc.in_features
        self.backbone = torch.nn.Sequential(*(list(self.backbone.children())[:-1]))

        # Additional hidden layers for further feature processing
        self.hidden_layers = nn.Sequential(
            nn.Linear(self.fc_in_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )
        
        self.sigmoid_activation = nn.Sigmoid()
        self.backbone.apply(self.initialize_weights)
        self.hidden_layers.apply(self.initialize_weights)
        self.fc_layers.apply(self.initialize_weights)

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_one(self, x):
        features = self.backbone(x)
        flattened_features = features.view(features.size()[0], -1)
        return flattened_features

    def forward(self, input1, input2):
        features1 = self.forward_one(input1)
        features2 = self.forward_one(input2)

        # Additional feature processing
        features1 = self.hidden_layers(features1)
        features2 = self.hidden_layers(features2)
        
        concatenated_features = torch.cat((features1, features2), 1)
        final_output = self.fc_layers(concatenated_features)
        final_output = self.sigmoid_activation(final_output)
        return final_output

