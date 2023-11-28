import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

def get_transform():
    """
    Constructs and returns a composed transform to be applied on the dataset samples.
    
    Returns:
    - transform (transforms.Compose): A composed transform consisting of converting images and masks to tensors and resizing them.
    """
    return transforms.Compose([
        transforms.Lambda(lambda sample: {'image': transforms.ToTensor()(sample['image']),
                                          'mask': transforms.ToTensor()(sample['mask'])}),
        transforms.Lambda(lambda sample: {'image': transforms.Resize((128, 128), antialias=True)(sample['image']),
                                          'mask': transforms.Resize((128, 128), antialias=True)(sample['mask'])}),
    ])

class ISICDataset(Dataset):
    """
    ISICDataset: Custom dataset class to handle ISIC data loading and preprocessing.
    
    Inherits from PyTorch's Dataset class, and overrides the methods:
    - __len__ to return the size of the dataset.
    - __getitem__ to get a sample from the dataset given an index.
    """
    def __init__(self, dataset_type, transform=None):
        """
        Initialize the ISICDataset instance.
        
        Parameters:
        - dataset_type (str): Indicates the type of dataset ('training', 'validation', or 'test').
        - transform (callable, optional): Optional transform to be applied on a sample.
        """
        assert dataset_type in ['training', 'validation', 'test'], "Invalid dataset type. Must be one of ['training', 'validation', 'test']"
        self.root_dir = os.path.join('../../../Data', dataset_type)
        mask = dataset_type + "_mask"
        self.maskdir = os.path.join('../../../Data', mask)
        self.transform = transform

        """ sort by masks and lesion images """
        self.image_list = sorted([f for f in os.listdir(self.root_dir) if f.endswith('.jpg')])
        self.mask_list = sorted([f for f in os.listdir(self.maskdir) if f.endswith('.png')])

    def __len__(self):
        """
        Returns the size/length of the dataset.
        
        Returns:
        - int: Number of samples in the dataset.
        """
        return len(self.image_list)

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset given an index. Lesion images as RGB, masks as greyscale.
        
        Parameters:
        - idx (int): Index of the sample to fetch.
        
        Returns:
        - sample (dict): A dictionary containing the image and mask.
        """
        img_name = os.path.join(self.root_dir, self.image_list[idx])
        mask_name = os.path.join(self.maskdir, self.mask_list[idx])
        
        image = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name).convert('L')
        
        sample = {'image': image, 'mask': mask}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample  # Returning a dict instead of a tuple