import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
import numpy as np


standardTransform = Compose([ToTensor(), Resize([128, 128], antialias=True), Normalize(0.5, 0.5 ,0.5)])
standardTransform = Compose([ToTensor(), Resize([128, 128], antialias=True)])

class OASISDataset(Dataset):  
    def __init__(self, img_data, transform=ToTensor()):
        self.imgs = img_data
        self.transform = transform
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        return self.transform(self.imgs[idx])

def load_images_from_directories(dirs, verbose=False):
  """Load images from data directory

  Args:
      dirs (_type_): _description_
  """
  image_paths = []
  images = []
  for dir in dirs:
      for filename in glob(dir, recursive=True):
          image_paths.append(filename)
          images.append(Image.open(filename))
  image_paths = sorted(image_paths)
  
  if verbose:
    for ip in image_paths[:10]:
        im = Image.open(ip)
        plt.figure()
        plt.imshow(im, cmap="gray")
  
  return image_paths, images


def load_oasis_images(verbose=False):
  """Load oasis images from data directory provided
  
  Returns:
      [string, list[np.array]]: image paths and images
  """
  dataset_names = ["test", "validate", "test"]
  dirs = []
  
  for dataset in dataset_names:
    dirs.append(f'./data/keras_png_slices_data/keras_png_slices_{dataset}/*')
    
  return load_images_from_directories(dirs, verbose)


  