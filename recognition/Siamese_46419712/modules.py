from imp import init_frozen
import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F


class SiameseModel(nn.Module):

    def __init__(self):
        super(SiameseModel, self).__init__()
        # Follow https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf -> Siamese Neural Networks for One-shot Image Recognition
        # first convolution layer
        self.model1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=10),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))

        self.model2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.model3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.model4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(6 * 6 * 256, 4096),
            nn.Sigmoid()
        )

        self.final_connect_layer = nn.Sequential(
            nn.Linear(4096, 1),
            nn.Sigmoid()
        )

    def init_forward(self, x):
        output = self.model1(x)
        output = self.model2(output)

        output = self.model3(output)

        output = self.model4(output)

        return output
    
    def forward(self, img1, img2):
        output1 = self.init_forward(img1)
        output2 = self.init_forward(img2)
        
        output = torch.abs(output1 - output2)
        output = self.final_connect_layer(output)

        return output