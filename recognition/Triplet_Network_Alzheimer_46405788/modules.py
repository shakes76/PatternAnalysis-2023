import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    A building block for a Residual Neural Network (ResNet).

    This class defines a single residual block, which is a fundamental component
    of the ResNet architecture. A residual block consists of two convolutional
    layers and shortcut connections. It is used to enable the training of deep
    neural networks while mitigating the vanishing gradient problem.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int, optional): The stride for the first convolutional layer. Default is 1.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) 
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        shortcut = self.shortcut(x)
        out += shortcut
        out = self.relu(out)
        
        return out
    
class ResNet(nn.Module):
    """
    A Residual Neural Network (ResNet) architecture.

    ResNet is known for its deep architecture with residual blocks that help 
    alleviate the vanishing gradient problem, allowing for the training of 
    very deep neural networks.

    Args:
        block (nn.Module): The type of residual block to use (e.g., ResidualBlock).
        num_blocks (list of int): A list specifying the number of blocks for each
            of the four ResNet layers.
        num_classes (int, optional): The number of output classes for classification.
            Default is 10 for general purposes.

    Attributes:
        in_channels (int): The number of input channels for the initial convolutional
            layer.

    Methods:
        make_layer(block, out_channels, num_blocks, stride):
            Creates a sequence of residual blocks for a ResNet layer.

    Forward Usage:
        To forward propagate input data through the network, call this class as a
        callable object.

    Example:
        resnet = ResNet(ResidualBlock, [2, 2, 2, 2], num_classes=2)
        outputs = resnet(input_data)
    """
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    def make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
    
class TripletLoss(nn.Module):
    """
    Triplet Loss for Siamese Networks or Metric Learning.

    Triplet loss is a loss function commonly used in Siamese networks or metric
    learning tasks. It encourages the model to minimize the distance between
    anchor and positive examples while maximizing the distance between anchor
    and negative examples by a specified margin.

    Args:
        margin (float): The margin parameter that defines the minimum acceptable
            difference between the distance of anchor-positive and anchor-negative
            pairs.

    Methods:
        forward(anchor, positive, negative, size_average=True):
            Computes the triplet loss given anchor, positive, and negative samples.

    Forward Usage:
        To compute the triplet loss, call this class as a callable object with
        anchor, positive, and negative samples as input.

    Example:
        triplet_loss = TripletLoss(margin=0.5)
        loss_value = triplet_loss(anchor, positive, negative)
    """
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

class TripletNet(nn.Module):
    """
    A Triplet Network for Siamese Networks or Metric Learning.

    The TripletNet class defines a neural network for Siamese networks or metric
    learning, specifically designed to generate embeddings for anchor, positive, and
    negative samples. It utilizes a ResNet-based architecture to produce feature vectors
    for the input samples.

    Methods:
        forward_one(x):
            Forward pass through the network for a single input.

        forward(anchor, positive, negative):
            Forward pass through the network for anchor, positive, and negative samples,
            generating embeddings for each.

    Example:
        triplet_net = TripletNet()
        anchor_embed, positive_embed, negative_embed = triplet_net(anchor, positive, negative)
    """
    def __init__(self):
        super(TripletNet, self).__init__()
        self.Resnet = ResNet(ResidualBlock, [2,2,2,2], 128)

    def forward_one(self, x):
        x = self.Resnet(x)
        return x

    def forward(self, anchor, positive, negative):
        output1 = self.forward_one(anchor)
        output2 = self.forward_one(positive)
        output3 = self.forward_one(negative)
        return output1, output2, output3
    
class TripletNetClassifier(nn.Module):
    def __init__(self):
        super(TripletNetClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
        
    def forward(self, x):
        x = self.classifier(x)
        return x