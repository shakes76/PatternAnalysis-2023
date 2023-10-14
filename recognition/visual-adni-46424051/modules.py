##################################   modules.py   ##################################

import torch
from torch.nn import Module, Linear, Parameter
from utils import patch, position

class Model(Module):
    def __init__(self, shape=(1, 105, 105), patches=7, hidden_dim=8):
        super(Model, self).__init__()
        self.shape = shape
        self.patches = patches
        self.hidden_dim = hidden_dim

        self.input_dim = int(shape[0] * self.patches ** 2)
        self.linear = Linear(self.input_dim, self.hidden_dim)

        self.token = Parameter(torch.rand(1, self.hidden_dim))

        self.pos = Parameter(torch.tensor(position(self.patches ** 2 + 1, self.hidden_dim)))
        self.pos.requires_grad = False
    
    def forward(self, img):
        patches = patch(img, self.patches)
        
        tokens = self.linear(patches)
        tokens = torch.stack([torch.vstack((self.token, tokens[i])) for i in range(len(tokens))])
        
        out = tokens + self.pos
        return out