import os
import torch
from torch.utils.data import Dataset
from PIL import Image

'''
    custom dataset for ISIC 17/18 Melanoma
'''
class ISICDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]  # TODO only .jpg files
        
        self.mask_dir = mask_dir
        self.mask_files = os.listdir(mask_dir)
        
        self.transform = transform
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        im_path = os.path.join(self.img_dir, self.img_files[idx])
        im = Image.open(im_path)
        
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        mask = Image.open(mask_path)
        
        if self.transform:
            im = self.transform(im)
            # mask = self.transform(mask)
            
        return im, mask
    
'''
    calculate mean and standard deviation
    modified from PyTorch documentation
'''
def calc_mean_std(loader):
    mean = 0.0
    std = 0.0
    n = 0

    for im, _ in loader:
        mean += im.sum(axis=[0,2,3])
        std += (im**2).sum(axis=[0,2,3])
        n += im.size(0)

    mean = mean/n
    std = torch.sqrt(std / n - (mean**2))

    return mean, std