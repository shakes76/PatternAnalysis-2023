import torch
import torch.nn as nn

class GCN(nn.Module):
    # input for command is the number of features in and the number of classes out
    def __init__(self, in_channels=128, num_classes=4):
        super(GCN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.layer1 = nn.Linear(in_channels, 256)
        self.layer2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, num_classes)
        # use xavier uniform as it is bettwen with deep neural networks with many layers and many nodes
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.zeros_(self.layer1.bias)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.zeros_(self.layer2.bias)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)
    
    def forward(self, x):
        z = torch.tanh(self.layer1(x))
        z = torch.tanh(self.layer2(z))
        z = self.out(z)
        # no softmax as we will be using cross entropy loss
        

