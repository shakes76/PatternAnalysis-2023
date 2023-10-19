import os
import zipfile
import torch

def accuracy(model, criterion, preds, truths):
  model.eval()
  preds = (preds > 0.5).float()
  correct = (preds==truths).sum()
  pixels = torch.numel(preds)
  dice_score = criterion(preds, truths.float())
  accuracy = correct / pixels*100
  avg_acc = 0
  print(accuracy.shape)

  return accuracy, dice_score.item()

