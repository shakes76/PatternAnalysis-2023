import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.init as init
import torch


class SiameseResNet(nn.Module):
    def __init__(self):
        super(SiameseResNet, self).__init__()

        resnet = models.resnet18(pretrained=True)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.embedding = nn.Sequential(*list(resnet.children())[:-1])

    def forward_one(self, x):
        # Forward pass for one input
        x = self.embedding(x)
        # Flatten the tensor
        x = x.view(x.size()[0], -1)
        return x

    def forward(self, x1, x2):
        # Forward pass for both inputs
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)

        # Calculate Euclidean Distance
        dist = torch.sqrt(torch.sum((out1 - out2) ** 2, dim=1))

        return dist

class ClassifierNet(nn.Module):
    def __init__(self, siamese_model):
        super(ClassifierNet, self).__init__()

        self.siamese_model = siamese_model
        for param in self.siamese_model.parameters():
            param.requires_grad = False  # Freeze Siamese model parameters

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),  # Assuming embedding size is 512 from ResNet18
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        embedding = self.siamese_model.forward_one(x)
        prob = self.classifier(embedding)
        return prob



class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function, assuming distance output from Siamese network.
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        # Compute contrastive loss
        loss_contrastive = torch.mean((label) * torch.pow(distance, 2) +
                                      (1-label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        return loss_contrastive




        

# def get_embedding_size(model, input_shape):
#     dummy_input = torch.randn(1, *input_shape)
#     dummy_output = model.forward_one(dummy_input)
#     return dummy_output.size(1)  # get the size of the output embedding for a single image

# # Example usage:
# siamese = SiameseResNet()
# embedding_size = get_embedding_size(siamese, (1,240,256))  # assuming your images are 224x224 grayscale
# print(embedding_size)


