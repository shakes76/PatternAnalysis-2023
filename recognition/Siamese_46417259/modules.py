import torch
import torch.nn as nn
import torch.nn.functional as F

# images are size [3, 240, 240]
class SiameseTwin(nn.Module):
    """
    the backbone / embedding network of the Siamese Neural Network
    expected input: images of size [3, 240, 240]
    output: feature vector of size [4096, 1]
    """
    def __init__(self) -> None:
        super(SiameseTwin, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 10, 1)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, 7, 1)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(128, 128, 4, 1)
        self.maxpool3 = nn.MaxPool2d(2, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, 1)
        self.fc = nn.Linear(256*22*22, 4096)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.maxpool1(out)
        out = F.relu(self.conv2(out))
        out = self.maxpool2(out)
        out = F.relu(self.conv3(out))
        out = self.maxpool3(out)
        out = F.relu(self.conv4(out))

        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
    
class SiameseNeuralNet(nn.Module):
    """
    the overall Siamese Neural Network
    expected input: two images of size [3, 240, 240] each
    output: two feature vectors of size [4096, 1] each
    """
    def __init__(self) -> None:
        super(SiameseNeuralNet, self).__init__()

        self.backbone = SiameseTwin()

    def forward(self, x1, x2):
        x1_features = self.backbone(x1)
        x2_features = self.backbone(x2)
        return x1_features, x2_features
    
    def get_backbone(self):
        return self.backbone

class SimpleMLP(nn.Module):
    """
    the classifier network
    works in conjunction with the embedding network
    expected input: feature vector of size [4096, 1] (the embedding network's output)
    output: a single value between 0 and 1 representing the likelihood an image is Cognitive Normal
    """
    def __init__(self) -> None:
        super(SimpleMLP, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.mlp(x)
        return out

#
# testing scripts
#
def test_one_twin():
    test = SiameseTwin()
    print(test)

    input = torch.rand(2, 3, 240, 240)
    x = test(input)
    print(x.shape)
    print(x)

def test_entire_net():
    net = SiameseNeuralNet()
    print(net)
    input1 = torch.rand(2, 3, 240, 240)
    input2 = torch.rand(2, 3, 240, 240)
    x, y = net(input1, input2)
    print(x.shape, y.shape)
    print(x)

if __name__ == "__main__":
    test_one_twin()
    test_entire_net()