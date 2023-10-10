import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
import numpy as np
from dataset import *

paths, imgs = load_oasis_images(verbose = False)
# dataset = OASISDataset(imgs, standardTransform)
# plt.imshow(imgs[0], cmap = 'gray')
# plt.show()

dataset = OASISDataset(imgs, standardTransform)

dl = DataLoader(dataset=dataset, batch_size=10, shuffle=False)
nextimg = next(iter(dl))
print(nextimg)
for img in nextimg:
  plt.imshow(imgs[0], cmap = 'gray')
  plt.show()