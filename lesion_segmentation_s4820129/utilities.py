import os
import zipfile
import torch

def accuracy(model, loader, device, criterion):
  model.eval()
  correct = 0
  pixels = 0
  dice_score = 0
  with torch.no_grad(): 
    for _,(preds, masks) in enumerate(loader):
      preds = preds.to(device)
      masks = masks.to(device).unsqueeze(1)
      preds = torch.sigmoid(preds)
      preds = (preds > 0.5).float()
      correct += (preds==masks).sum()
      pixels += torch.numel(preds)
      dice_score += criterion(preds, masks.float())
  
  dice_score = dice_score/len(loader)
  accuracy = correct / pixels*100
  print(
    f'Accuracy: {accuracy:.2f}, Dice Similarity Coefficient: {dice_score:.2f}, Loss: {1-dice_score:.2f}'
  )
  return accuracy, dice_score

