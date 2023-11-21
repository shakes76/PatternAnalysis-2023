import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.init as init
import torch

# Referenced from: https://github.com/pytorch/examples/blob/main/siamese_network/main.py
class SiameseResNet(nn.Module):
    def __init__(self):
        super(SiameseResNet, self).__init__()

        # Use pytorch's resnet 18 as the backbone
        resnet = models.resnet18(pretrained=True)
        # Modify the first convolutional layer channel
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
         # Remove the last layer of resnet18
        self.embedding = nn.Sequential(*list(resnet.children())[:-1])

    def forward_one(self, x):
        # Forward pass for one input
        x = self.embedding(x)
        x = x.view(x.size()[0], -1)
        return x

    def forward(self, x1, x2):
        # Forward pass for both inputs
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)

        # Calculate Euclidean Distance
        dist = torch.sqrt(torch.sum((out1 - out2) ** 2, dim=1))

        return dist


# Referenced from: ChatGPT (Version 4) Query: make a classifier with my given siamese network
# This is the classifer architecture that gives the highest accuracy on the test set
class ClassifierNet(nn.Module):
    def __init__(self, siamese_model):
        super(ClassifierNet, self).__init__()

        self.siamese_model = siamese_model
        for param in self.siamese_model.parameters():
            param.requires_grad = False  # Freeze Siamese model parameters

        # Classifier head with increased complexity
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),  
            nn.ReLU(),
            nn.BatchNorm1d(512), 
            nn.Dropout(0.7),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),  
            nn.Dropout(0.7),

            nn.Linear(256, 128),  
            nn.ReLU(),
            nn.BatchNorm1d(128),  
            nn.Dropout(0.7),

            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        embedding = self.siamese_model.forward_one(x)
        prob = self.classifier(embedding)
        return prob


# Referenced from: https://pyimagesearch.com/2021/01/18/contrastive-loss-for-siamese-networks-with-keras-and-tensorflow/
class ContrastiveLoss(torch.nn.Module):
   
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        # Compute contrastive loss
        loss_contrastive = torch.mean((label) * torch.pow(distance, 2) +
                                      (1-label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        return loss_contrastive






