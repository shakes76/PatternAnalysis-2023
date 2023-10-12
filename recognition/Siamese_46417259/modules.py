import torch
import torch.nn as nn
# import torchvision
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
        # out = F.sigmoid(self.fc(out))
        return out
    
class SiameseNeuralNet(nn.Module):
    def __init__(self) -> None:
        super(SiameseNeuralNet, self).__init__()

        self.backbone = SiameseTwin()
        self.fc = nn.Linear(4096, 1)

    def forward(self, x1, x2):
        x1_features = self.backbone(x1)
        x2_features = self.backbone(x2)
        # out = F.pairwise_distance(x_features, y_features, keepdim=True)
        # out = torch.absolute(pairwise_subtraction(x_features, y_features))
        out = torch.absolute(x1_features - x2_features)
        out = F.sigmoid(self.fc(out))
        return out, x1_features, x2_features
    
    def get_backbone(self):
        return self.backbone
    
class SiameseMLP(nn.Module):
    
    def __init__(self, backbone: SiameseTwin) -> None:
        super(SiameseMLP, self).__init__()
        
        self.backbone = backbone
        self.backbone.eval()
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
        # get the feature vector from the siamese twin
        self.backbone.eval()
        with torch.no_grad():
            out = self.backbone(x) # size [{batch_size}, 4096]
        out = self.mlp(out)
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
    input1 = torch.rand(2, 3, 240, 256)
    input2 = torch.rand(2, 3, 240, 256)
    x = net(input1, input2)
    print(x.shape)
    print(x)

def test_mlp():
    backbone = SiameseTwin()
    mlp = SiameseMLP(backbone)

    print("num of parameters overall: ", sum([param.nelement() for param in mlp.parameters()]))
    print("num of mlp params: ", sum([param.nelement() for param in mlp.mlp.parameters()]))

    input1 = torch.rand(2, 3, 240, 240)
    out = mlp(input1)
    print(out.shape)
    print(out)

    out = out.view(-1)
    print(out.shape)
    print(out)

    criterion = nn.BCELoss()
    label = torch.tensor([1, 0]).float()
    loss = criterion(out, label)
    print(loss)

    predicted = (out > 0.5).float()
    print(predicted)
    # _, predicted = torch.max(outputs.data, 1)
    total = label.size(0)
    correct = (predicted == label).sum().item()
    print(total, correct)

    # input = torch.randn(3, 2, requires_grad=True)
    # target = torch.rand(3, 2, requires_grad=False)
    # loss = F.binary_cross_entropy(torch.sigmoid(input), target)
    # print(loss)




if __name__ == "__main__":
    test_one_twin()
    # test_entire_net()
    # test_mlp()



#
# deprecated code
#
def pairwise_subtraction(x:torch.Tensor, y:torch.Tensor):
    if x.shape != y.shape:
        raise NotImplementedError("X and Y must be the same shape")
    pass

    out = torch.zeros(x.shape)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            out[i,j] = x[i,j] - y[i,j]
    return out

    # return torch.as_tensor([a - b for a,b in zip(x,y)])

def test_pairwise_subtraction():
    x = torch.rand(2, 4)
    y = torch.rand(2, 4)
    print(x)
    print(y)
    print(pairwise_subtraction(x, y))
    print(F.pairwise_distance(x, y))

def test_pairwise_subtraction_unequal_shape():
    x = torch.rand(1, 4)
    y = torch.rand(2, 4)
    print(x)
    print(y)
    print(pairwise_subtraction(x, y))

# test_pairwise_subtraction_unequal_shape()
# test_pairwise_subtraction()
