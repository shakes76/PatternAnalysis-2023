import os
import zipfile
import torch

def accuracy(preds, truths):
  with torch.no_grad():
    preds = (preds > 0.5).float()
    correct = (preds==truths).sum()
    pixels = torch.numel(preds)
    accuracy = correct / pixels + 1e-8

  return accuracy

#compute mean and standard deviation of the custom dataset
def get_statistics(dataset):
  image_mean_R = 0
  image_mean_B = 0
  image_mean_G = 0
  image_std_R = 0
  image_std_B = 0
  image_std_G = 0

  mask_mean = 0
  mask_std = 0
  for idx in range(len(dataset)):
    print(idx)
    image, mask = dataset[idx]
    image = image.to('cuda')
    mask = image.to('cuda')
    out = torch.mean(image, dim=[1,2])
    image_mean_R += out[0] 
    image_mean_B += out[1]
    image_mean_G += out[2] 
    out = torch.std(image, dim=[1,2])
    image_std_R += out[0]
    image_std_B += out[1]
    image_std_G += out[2]
    mask_mean += torch.mean(mask, dim=[1,2])
    mask_std += torch.std(mask, dim=[1,2])

  return [image_mean_R/len(dataset), image_mean_G/len(dataset), image_mean_B/len(dataset)], [image_std_R/len(dataset), image_std_B/len(dataset), image_std_G/len(dataset)], mask_mean/len(dataset), mask_std/len(dataset)


