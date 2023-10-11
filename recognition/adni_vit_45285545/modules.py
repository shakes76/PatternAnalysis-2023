'''
Components of ViT model.
'''
import torch.nn as nn
import torchvision

class ViT(nn.Module):
    def __init__(self):
        super(ViT, self).__init__()
        self.mdl = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.mdl.heads.head = nn.Linear(in_features=768, out_features=2, bias=True)

    def forward(self, x):
        return self.mdl.forward(x)
