import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class ADNIDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        # root_dir should be the path to the 'AD_NC' folder
        self.root_dir = root_dir
        self.transform = transform
        
        # Choose the directory based on whether it's training or testing data
        self.data_dir = os.path.join(root_dir, "train" if train else "test")
        
        self.ad_files = [os.path.join(self.data_dir, "AD", fname) for fname in os.listdir(os.path.join(self.data_dir, "AD"))]
        self.nc_files = [os.path.join(self.data_dir, "NC", fname) for fname in os.listdir(os.path.join(self.data_dir, "NC"))]
        
        # Combine AD and NC file paths into a single list
        self.files = self.ad_files + self.nc_files

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        
        # Apply transformations if any
        if self.transform:
            img = self.transform(img)

        return img

def get_data_loader(root_dir, train=True, transform=None, batch_size=32, shuffle=True):
    dataset = ADNIDataset(root_dir, train, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
