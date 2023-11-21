
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class TripletSiameseNetwork(nn.Module):
    def __init__(self, pretrained=True):
        super(TripletSiameseNetwork, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm1d(1000)
        self.fc1 = nn.Linear(1000, 128)

    def forward_once(self, x):
        output = self.resnet(x)
        output = self.bn(output)
        output = self.fc1(output)
        return output

    def forward(self, anchor, positive, negative):
        output_anchor = self.forward_once(anchor)
        output_positive = self.forward_once(positive)
        output_negative = self.forward_once(negative)
        return output_anchor, output_positive, output_negative

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive)
        distance_negative = F.pairwise_distance(anchor, negative)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

class TripletLossWithRegularization(nn.Module):
    def __init__(self, model, margin=1.0, lambda_reg=0.0001):
        super(TripletLossWithRegularization, self).__init__()
        self.model = model
        self.margin = margin
        self.lambda_reg = lambda_reg  # Regularization parameter

    def forward(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive)
        distance_negative = F.pairwise_distance(anchor, negative)
        triplet_losses = F.relu(distance_positive - distance_negative + self.margin)
        triplet_loss = triplet_losses.mean()

        # Compute L2 regularization
        l2_reg = None
        for param in self.model.parameters():
            if l2_reg is None:
                l2_reg = param.norm(2)
            else:
                l2_reg = l2_reg + param.norm(2)

        loss = triplet_loss + self.lambda_reg * l2_reg
        return loss
