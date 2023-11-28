"""
    File name: modules.py
    Author: Fanhao Zeng
    Date created: 11/10/2023
    Date last modified: 16/10/2023
    Python Version: 3.10.12
"""

import torch
import torch.nn as nn
import torchvision


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # get resnet model
        self.resnet = torchvision.models.resnet18(weights=None)

        # over-write the first conv layer to be able to read ADNI images
        # as resnet18 reads (3,x,x) where 3 is RGB channels
        # whereas ADNI has (1,x,x) where 1 is a gray-scale channel
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc_in_features = self.resnet.fc.in_features

        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.sigmoid = nn.Sigmoid()

        # initialize the weights
        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        # initialize the weights of the linear layers
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_one(self, x):
        # get the features of one image
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)

        # concatenate both images' features
        output = torch.cat((output1, output2), 1)

        # pass the concatenation to the linear layers
        output = self.fc(output)

        # pass the out of the linear layers to sigmoid layer
        output = self.sigmoid(output)

        return output


class ADNIClassifier(nn.Module):
    def __init__(self, siamese_network):
        super(ADNIClassifier, self).__init__()
        # Use Siamese Network's feature extractor
        self.feature_extractor = siamese_network.resnet

        # add linear layers to classify between AD and NC
        self.classifier = nn.Sequential(
            # Output from feature extractor to 1 neuron (binary classification)
            nn.Linear(siamese_network.fc_in_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output
