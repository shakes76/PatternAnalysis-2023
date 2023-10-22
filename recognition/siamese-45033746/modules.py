import torch.nn as nn

SIAMESE_FEATURES = 2

"""
modules.py

define triple siamese convolutional neural network and binary classifer neural net
"""

class SiameseNetwork(nn.Module):
    """
    Siamese CNN
    Reference: https://github.com/maticvl/dataHacker/blob/master/pyTorch/014_siameseNetwork.ipynb
    """

    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 32, 11, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 5, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1),
            nn.ReLU(inplace=True),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(386688, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, SIAMESE_FEATURES)
        )

    def forward_once(self, tensor):
        # This function will be called on each of the triplet in forward()
        # It's output is used to determine the similiarity
        # REDO
        cnn_output = self.cnn1(tensor)
        cnn_output = cnn_output.view(cnn_output.size()[0], -1)
        cnn_output = self.fc1(cnn_output)
        return cnn_output

    def forward(self, anchor, positive, negative):
        """
        Calls forward on all three images in CNN
        :param anchor: torch.Tensor
        :param positive: torch.Tensor
        :param negative: torch.Tensor
        :return:
        """
        # In this function we pass in  triplet images and obtain triplet vectors
        # which are returned
        # REDO
        anchor_vec = self.forward_once(anchor)
        positive_vec = self.forward_once(positive)
        negative_vec = self.forward_once(negative)

        return anchor_vec, positive_vec, negative_vec


class BinaryClassifier(nn.Module):
    """
    Binary Classifier Neural Net to discriminate between AD and NC
    """

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(SIAMESE_FEATURES, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
        )

    def forward(self, input):
        return self.fc1(input)
