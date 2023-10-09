'''
Components of ViT model.
'''
import torch.nn as nn
import torchvision

class ViT(nn.Module):
    def __init__(self):
        super(ViT, self).__init__()
        self.mdl = torchvision.models.vit_b_32()
        self.clf = nn.Linear(in_features=768, out_features=2, bias=True)
        self.mdl.heads.head = self.clf

    def forward(self, x):
        return self.mdl.forward(x)
