import os
import zipfile
import torch

def accuracy(model, criterion, preds, truths):
  model.eval()
  correct = (preds==truths).sum()
  pixels = torch.numel(preds)
  dice_score = criterion(preds, truths)
  accuracy = correct / pixels*100 + 1e-8

  return accuracy, dice_score.item()

