##################################   modules.py   ##################################
import torch
from torch.nn import Module, Sequential, Linear, Sigmoid, Conv2d, ReLU, MaxPool2d

class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = Sequential(
            Conv2d(1, 64, 10),
            ReLU(inplace=True),
            MaxPool2d(2),
            Conv2d(64, 128, 7),
            ReLU(),
            MaxPool2d(2),
            Conv2d(128, 128, 4),
            ReLU(),
            MaxPool2d(2),
            Conv2d(128, 256, 4),
            ReLU()
        )
        self.linear = Sequential(Linear(135168, 8192), Sigmoid())
        self.out = Linear(8192, 1)
    
    def forward_one(self, img):
        img = self.conv(img)
        img = img.view(img.size()[0], -1)
        return self.linear(img)
        
    def forward(self, img1, img2):
        out1 = self.forward_one(img1)
        out2 = self.forward_one(img2)
        return out1, out2