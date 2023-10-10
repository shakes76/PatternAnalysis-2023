import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import random

class TripletDataset(Dataset):
    """
    Generate triplets for training Siamese networks using triplet loss.
    
    For each anchor image from either the AD or NC class, a positive image is selected from 
    the same patient but a different slice. A negative image is then chosen from the opposite class.
    
    Args:
        root_dir (str): Root directory containing AD and NC image subdirectories.
        transform (callable, optional): Transformations applied to the images.
    
    Returns:
        tuple: A triplet of images - (anchor, positive, negative).
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Load all image paths
        self.ad_paths = [os.path.join(root_dir, 'train/AD', img) for img in os.listdir(os.path.join(root_dir, 'train/AD'))]
        self.nc_paths = [os.path.join(root_dir, 'train/NC', img) for img in os.listdir(os.path.join(root_dir, 'train/NC'))]

    def __len__(self):
        return min(len(self.ad_paths), len(self.nc_paths))

    def __getitem__(self, idx):
        # Extract patient ID from the filename
        patient_id = self.ad_paths[idx].split('/')[-1].split('_')[0]
        
        # Choose an anchor image
        anchor_path = self.ad_paths[idx]
        
        # Choose a positive image from the same patient
        positive_path = random.choice([path for path in self.ad_paths if path != anchor_path and patient_id in path])
        
        # Choose a negative image from a different patient
        negative_path = random.choice(self.nc_paths)

        anchor_image = Image.open(anchor_path)
        positive_image = Image.open(positive_path)
        negative_image = Image.open(negative_path)

        if self.transform:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)

        return anchor_image, positive_image, negative_image
