"""
modules.py: Siamese Network and Contrastive Loss for image similarity learning.

Author: Rachit Chaurasia (s4823870)
Date: 20/10/2023

This module defines a Siamese Network for image similarity learning and the Contrastive Loss
used for training the Siamese Network. The Siamese Network takes pairs of images as input and
produces embeddings for them. The Contrastive Loss calculates the loss between the embeddings
of the pairs.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    """
    Siamese Network for image similarity learning.

    This network takes pairs of images and produces embeddings for them.

    Args:
        None
    """
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 10),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 7),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 24 * 24, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.Linear(256, 2)
        )

    def forward_one(self, x):
        """
        Forward pass for a single input.

        Args:
            x (Tensor): Input image tensor.

        Returns:
            Tensor: Embedding for the input image.
        """
        x = self.cnn(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        """
        Forward pass for pairs of input images.

        Args:
            input1 (Tensor): First input image.
            input2 (Tensor): Second input image.

        Returns:
            Tuple: Pair of embeddings for the input images.
        """
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2

class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for Siamese Network training.

    Args:
        margin (float, optional): Margin value for the contrastive loss. Default is 2.0.
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        """
        Calculate the contrastive loss.

        Args:
            output1 (Tensor): Embedding for the first image.
            output2 (Tensor): Embedding for the second image.
            label (Tensor): Label (0 for dissimilar, 1 for similar).

        Returns:
            Tensor: Contrastive loss value.
        """
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive