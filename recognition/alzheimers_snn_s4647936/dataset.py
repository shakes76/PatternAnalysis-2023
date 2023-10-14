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
        mode (str): Either 'train' or 'test'.
        transform (callable, optional): Transformations applied to the images.
    
    Returns:
        tuple: A triplet of images - (anchor, positive, negative).
    """
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        
        # Directories for AD and NC images
        self.ad_dir = os.path.join(root_dir, mode, 'AD')
        self.nc_dir = os.path.join(root_dir, mode, 'NC')

        # Load all image paths
        self.ad_paths = [os.path.join(self.ad_dir, img) for img in os.listdir(self.ad_dir)]
        self.nc_paths = [os.path.join(self.nc_dir, img) for img in os.listdir(self.nc_dir)]

    def __len__(self):
        return len(self.ad_paths) + len(self.nc_paths)  # combined length

    def __getitem__(self, idx):
        # Decide whether to take AD or NC as anchor based on index
        if idx < len(self.ad_paths):
            anchor_path = self.ad_paths[idx]
            positive_dir = self.ad_dir
            negative_dir = self.nc_dir
        else:
            anchor_path = self.nc_paths[idx - len(self.ad_paths)]  # offset by length of ad_paths
            positive_dir = self.nc_dir
            negative_dir = self.ad_dir

        # Extract patient ID from the filename
        patient_id = anchor_path.split('/')[-1].split('_')[0]

        # Choose a positive image from the same patient
        positive_path = random.choice([path for path in os.listdir(positive_dir) if path != os.path.basename(anchor_path) and patient_id in path])
        positive_path = os.path.join(positive_dir, positive_path)  # complete path

        # Choose a negative image from a different patient
        negative_img = random.choice([img for img in os.listdir(negative_dir) if patient_id not in img])
        negative_path = os.path.join(negative_dir, negative_img)

        anchor_image = Image.open(anchor_path)
        positive_image = Image.open(positive_path)
        negative_image = Image.open(negative_path)

        if self.transform:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)

        return anchor_image, positive_image, negative_image
