from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import cv2
import numpy as np
import torch

# Dataset for training, validating, testing
class adniDataset(Dataset):

    def __init__(self, ad_dir, nc_dir, ads, ncs, transform=None):
        self.ad_dir = ad_dir # /home/groups/comp3710/ADNI/AD_NC/train/AD
        self.nc_dir = nc_dir # /home/groups/comp3710/ADNI/AD_NC/train/NC

        self.ads = sorted(ads)
        self.ncs = sorted(ncs)

        self.transform = transform

    def __len__(self):
        return len(self.ads) + len(self.ncs)

    def __getitem__(self, idx):
        # Determine if AD or NC 
        if idx < len(self.ads):
            path = os.path.join(self.ad_dir, self.ads[idx])
            label = 1
        else:
            path = os.path.join(self.nc_dir, self.ncs[idx-len(self.ads)])
            label = 0
        
        # Read image
        # Positive and negative images are randomly sampled
        image = cv2.imread(path)

        # Convert to np array and resize to single channel
        image = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

        # Normalise
        image = image / 255.0

        # Transformation
        if self.transform != None:
            image = self.transform(image)

        return image, label

# Dataset for embeddings 
class embeddingsDataset(Dataset):
   def __init__(self, tensor_path, device):
       self.tensors = os.listdir(tensor_path)
       self.dir = tensor_path
       self.device = device

   def __len__(self):
       return len(self.tensors)
   
   def __getitem__(self, idx):
       data = torch.load(f"{self.dir}/{self.tensors[idx]}", map_location='cpu')
       return data["embeddings"], data["labels"]

# Dataset for prediction
class adniPredictionDataset(Dataset):

    def __init__(self, dir, transform=None):
        self.dir = dir
        self.files = os.listdir(dir)
        if transform != None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([transforms.ToTensor(),
                             transforms.CenterCrop((240, 240)),
                             transforms.Resize((225, 225)),
                             transforms.Normalize(mean=0.0, std=1.0)])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.dir, self.files[idx])

        # Read image
        # Positive and negative images are randomly sampled
        image = cv2.imread(path)

        # Convert to np array and resize to single channel
        image = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

        # Normalise
        image = image / 255.0

        # Transformation
        image = self.transform(image)

        return image