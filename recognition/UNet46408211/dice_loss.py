import torch
import torch.nn as nn
import numpy as np

class DiceLossLogits(nn.Module):
    """
    A binary loss function based on the Dice score.
    """
    
    def __init__(self, smooth=1e-9):
        super().__init__()
        self._smooth = smooth
    
    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()
        dice = (2 * intersection + self._smooth) / (union + self._smooth)
        return 1 - dice