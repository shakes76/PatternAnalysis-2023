import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16
from torchvision.models import ViT_B_16_Weights

"""
ViT class. Includes vit_b_16 model with MLP head to suit the task (i.e. 2 output features, AD or NC)
"""
class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.model.heads.head = nn.Linear(in_features=768, out_features=2)

    def forward(self, x):
        return self.model.forward(x)