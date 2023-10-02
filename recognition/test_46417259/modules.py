import torch
import torch.nn as nn
import torchvision
# import torchvision.transforms.v2 as transforms
import torch.nn.functional as F

# images are size [3, 240, 256]
class SiameseTwin(nn.Module):

    def __init__(self) -> None:
        super(SiameseTwin, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 10, 1)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, 7, 1)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(128, 128, 4, 1)
        self.maxpool3 = nn.MaxPool2d(2, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, 1)
        self.fc = nn.Linear(256*22*24, 4096)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.maxpool1(out)
        out = F.relu(self.conv2(out))
        out = self.maxpool2(out)
        out = F.relu(self.conv3(out))
        out = self.maxpool3(out)
        out = F.relu(self.conv4(out))

        out = torch.flatten(out, 1)
        out = F.sigmoid(self.fc(out))
        return out
    

# testing
def test_one_twin():
    test = SiameseTwin()
    print(test)

    input = torch.rand(1, 3, 240, 256)
    x = test(input)
    print(x.shape)

test_one_twin()