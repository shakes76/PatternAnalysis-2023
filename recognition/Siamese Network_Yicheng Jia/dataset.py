import torch
import torch.nn as nn
import torchvision


def init_weights(m):
    # initialize the weights of the linear layers
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # get resnet model
        self.resnet = torchvision.models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc_in_features = self.resnet.fc.in_features

        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 256),
            nn.BatchNorm1d(256),  # Batch Normalization
            nn.ReLU(inplace=True),  # Activate function
            nn.Dropout(p=0.2),  # Dropout
            nn.Linear(256, 1),
        )

        self.sigmoid = nn.Sigmoid()  # Convert the range to [0,1]

        # initialize the weights
        self.resnet.apply(init_weights)
        self.fc.apply(init_weights)

    def forward_one(self, x):
        # get the features of one image
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)

        # concatenate both images' features
        output = torch.cat((output1, output2), 1)

        # pass the concatenation to the linear layers
        output = self.fc(output)

        # pass the out of the linear layers to sigmoid layer
        output = self.sigmoid(output)

        return output
