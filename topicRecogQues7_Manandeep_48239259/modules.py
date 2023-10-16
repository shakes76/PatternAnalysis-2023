import torch
import torch.nn as tnn
import torch.nn.functional as TorchFun

class CustomSiameseNetwork(tnn.Module):
    def __init__(self):
        super(CustomSiameseNetwork, self).__init__()
        self.cnn = tnn.Sequential(
            tnn.ReflectionPad2d(1),
            tnn.Conv2d(1, 4, kernel_size=3),
            tnn.ReLU(inplace=True),
            tnn.BatchNorm2d(4),
            tnn.ReflectionPad2d(1),
            tnn.Conv2d(4, 8, kernel_size=3),
            tnn.ReLU(inplace=True),
            tnn.BatchNorm2d(8),
            tnn.ReflectionPad2d(1),
            tnn.Conv2d(8, 8, kernel_size=3),
            tnn.ReLU(inplace=True),
            tnn.BatchNorm2d(8),
        )

        self.fc1 = tnn.Sequential(
            tnn.Linear(8 * 100 * 100, 500),
            tnn.ReLU(inplace=True),
            tnn.Linear(500, 500),
            tnn.ReLU(inplace=True),
            tnn.Linear(500, 5)
        )

    def forward_once(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = TorchFun.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
