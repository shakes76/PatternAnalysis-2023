import torch.nn as nn
import torchvision.models as models


class SiameseResNet(nn.Module):
    def __init__(self):
        super(SiameseResNet, self).__init__()
        # Use pretrained ResNet50 as the backbone of our model.
        self.resnet = models.resnet50(pretrained=True)
        """
       The last layer of ResNet50 is a fully connected layer.
        We need to remove the fully connected layer, and we have to use our own classifier. 
        This is because the fully connected layer is designed to classify 1000 classes, but in our case, we only have 2 classes.
        """
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        # Add our own fully connected layer as the classifier.

        self.fc = nn.Sequential(
            # Flatten the output of the previous layer.
            nn.Flatten(),

            # The input size is 2048 because the output of ResNet50 is a 2048-dimensional vector.
            nn.Linear(2048, 1024),

            # ReLU activation function
            nn.ReLU(inplace=True),

            # Dropout layer to prevent overfitting.
            nn.Dropout(0.5),

            # The input size is 1024 because the output of the previous layer is a 1024-dimensional vector.
            nn.Linear(1024, 512),

            # ReLU activation function
            nn.ReLU(inplace=True),

            # Dropout layer to prevent overfitting.
            nn.Dropout(0.5),

            # The input size is 512 because the output of the previous layer is a 512-dimensional vector.
            nn.Linear(512, 2)
        )

    # Forward pass for a single image.
    def forward_image_single(self, x):
        # Pass the image through the ResNet backbone.
        x = self.features(x)
        # Pass the output of ResNet through our classifier.
        x = self.fc(x)
        return x

    # Forward pass for a pair of images. This is the main function of siamese network.
    def forward_image_pairs(self, input1, input2):
        # Pass the first image through the network.
        output1 = self.forward_image_single(input1)
        # Pass the second image through the network.
        output2 = self.forward_image_single(input2)
        return output1, output2
