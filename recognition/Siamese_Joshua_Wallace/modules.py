import torch
import torch.nn as nn
import torch.nn.functional as F


class Siamese(nn.Module):

    def __init__(self):
        super(Siamese, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=10), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=7), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=4), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=4), nn.ReLU()
        )

        self.embedding = nn.Sequential(
            nn.Linear(128*24*24, 512),
            nn.Linear(512, 1),

            nn.ReLU()
        )

        # Classify whether the images belong to the same class or different classes
        self.fc = nn.Sequential(
            nn.Sigmoid()
        )

        # self.conv1 = nn.Conv2d(3, 64, 10)
        # self.conv2 = nn.Conv2d(64, 128, 7)
        # self.conv3 = nn.Conv2d(128, 256, 4)
        # self.conv4 = nn.Conv2d(256, 512, 4)

        # self.pool1 = nn.MaxPool2d(2, stride=2)
        # self.pool2 = nn.MaxPool2d(2, stride=2)
        # self.pool3 = nn.MaxPool2d(2, stride=2)
        # self.pool4 = nn.MaxPool2d(2, stride=2)

        # self.relu1 = nn.ReLU(inplace=True)
        # self.relu2 = nn.ReLU(inplace=True)
        # self.relu3 = nn.ReLU(inplace=True)
        # self.relu4 = nn.ReLU(inplace=True)

        # self.fc1 = nn.Linear(128*24*24, 512)
        # self.fc2 = nn.Linear(512, 1)

        # self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        h1 = self.sub_forward(x1)
        h2 = self.sub_forward(x2)
        diff = torch.abs(h1 - h2)
        scores = self.fc(diff)
        
        return scores

    def sub_forward(self, x):
        x = self.conv(x)
        x = self.embedding(x)
        return x
    
class Residual(nn.Module):
    def __init__(self, in_channels, n_hidden, n_residual):
        super(Residual, self).__init__()

        self.residual = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=n_hidden, 
                out_channels=n_residual, 
                kernel_size=3, 
                stride=1, 
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(in_channels=n_residual, 
                out_channels=n_hidden, 
                kernel_size=1, 
                stride=1, 
                padding=0
            ),
        )
    
    def forward(self, out):
        return out + self.residual(out)


class Encoder(nn.Module):
    """
    Encoder module for the VQ-VAE model.

    The encoder consists of 2 strided convolutional layers with stride 2 and 
    window size 4 × 4, followed by two residual 3 × 3 blocks (implemented as 
    ReLU, 3x3 conv, ReLU, 1x1 conv), all having 256 hidden units.
    """

    def __init__(self, in_channels, n_hidden, n_residual):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=n_hidden // 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            in_channels=n_hidden // 2, 
            out_channels=n_hidden, 
            kernel_size=4, 
            stride=2, 
            padding=1
        )

        self.residual1 = Residual(n_hidden, n_hidden, n_residual)
        self.residual2 = Residual(n_hidden, n_hidden, n_residual)

        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, out):
        out = self.conv1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.residual1(out)
        out = self.residual2(out)
        out = self.relu2(out)
        return out
    
class Decoder(nn.Module):
    """
    Decoder
    """
    def __init__(self, in_channels, n_hidden, n_residual):
        super(Decoder, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=n_hidden,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.residual1 = Residual(n_hidden, n_hidden, n_residual)
        self.residual2 = Residual(n_hidden, n_hidden, n_residual)

        self.relu = nn.ReLU()

        self.transpose1 = nn.ConvTranspose2d(
            in_channels=n_hidden,
            out_channels=n_hidden // 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        
        self.transpose2 = nn.ConvTranspose2d(
            in_channels=n_hidden // 2,
            out_channels=1,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, out):
        out = self.conv1(out)
        out = self.residual1(out)
        out = self.residual2(out)
        out = self.relu(out)
        out = self.transpose1(out)
        out = self.transpose2(out)
        return out


