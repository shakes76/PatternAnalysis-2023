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