##################################   modules.py   ##################################
import torch
from torch.nn import Module, Sequential, Linear, Sigmoid, Conv2d, ReLU, MaxPool2d, Dropout2d, LocalResponseNorm

class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = Sequential(
            Conv2d(1, 96, 11),
            ReLU(inplace=True),
            LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            MaxPool2d(3),
            Dropout2d(p=0.3),
            Conv2d(96, 256, 5),
            ReLU(inplace=True),
            LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            MaxPool2d(3),
            Dropout2d(p=0.3),
            Conv2d(256, 384, 3),
            ReLU(inplace=True),
            Conv2d(384, 256, 3),
            ReLU(inplace=True),
            MaxPool2d(3),
            Dropout2d(p=0.3),
        )
        self.linear = Sequential(
            Linear(256, 2)
        )
    
    def forward_one(self, img):
        img = self.conv(img)
        img = img.view(img.size()[0], -1)
        return self.linear(img)
        
    def forward(self, img1, img2):
        out1 = self.forward_one(img1)
        out2 = self.forward_one(img2)
        return out1, out2