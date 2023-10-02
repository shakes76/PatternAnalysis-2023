import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class SiameseResNet(nn.Module):
    def __init__(self):
        super(SiameseResNet, self).__init__()
        # Use pretrained ResNet18 as the backbone of our model.
        self.resnet = models.resnet18(weights=None)  # Use the pretrained flag to load the pretrained weights
        """
        The last layer of ResNet18 is a fully connected layer.
        We need to remove the fully connected layer, and we have to use our own classifier. 
        This is because the fully connected layer is designed to classify 1000 classes, but in our case, we only have 2 classes.
        """
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        # Add our own fully connected layer as the classifier.

        self.fc = nn.Sequential(
            # Flatten the output of the previous layer.
            nn.Flatten(),

            # The input size is 512 because the output of ResNet18 is a 512-dimensional vector.
            nn.Linear(512, 256),

            # Batch normalization layer to normalize the output of the previous layer.
            nn.BatchNorm1d(256),

            # ReLU activation function
            nn.ReLU(inplace=True),

            # Dropout layer to prevent overfitting.
            nn.Dropout(0.4),

            # The input size is 256 because the output of the previous layer is a 256-dimensional vector.
            nn.Linear(256, 128),

            # Batch normalization layer to normalize the output of the previous layer.
            nn.BatchNorm1d(128),

            # ReLU activation function
            nn.ReLU(inplace=True),

            # Dropout layer to prevent overfitting.
            nn.Dropout(0.4),

            # The input size is 128 because the output of the previous layer is a 128-dimensional vector.
            nn.Linear(128, 2),
        )

    # Forward pass for a single image.
    def forward_image_single(self, x):
        # Pass the image through the ResNet backbone.
        x = self.features(x)
        # Pass the output of ResNet through our classifier.
        x = self.fc(x)
        return x

    # Forward pass for a pair of images. This is the main function of siamese network.
    def forward(self, input1, input2):
        # Pass the first image through the network.
        output1 = self.forward_image_single(input1)
        # Pass the second image through the network.
        output2 = self.forward_image_single(input2)
        return output1, output2


class SiameseVGG(nn.Module):
    def __init__(self):
        super(SiameseVGG, self).__init__()
        # Use pretrained VGG16 as the backbone of our model.
        self.vgg = models.vgg16(pretrained=True)  # Use the pretrained flag to load the pretrained weights
        self.features = self.vgg.features
        # Add our own fully connected layer as the classifier.
        self.fc = nn.Sequential(
            # Flatten the output of the previous layer.
            nn.Flatten(),
            # The input size is 512*7*7 because the output of VGG16 is a 512x7x7 feature map.
            nn.Linear(512*7*7, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 2),
        )

    def forward_image_single(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_image_single(input1)
        output2 = self.forward_image_single(input2)
        return output1, output2
