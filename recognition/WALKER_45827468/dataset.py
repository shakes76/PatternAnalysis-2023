import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
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
    
    def find_mask(self, im):
        im_num = os.path.splitext(im)[0]
        mask = im_num + '_segmentation.png'
        return mask
    
    def __getitem__(self, idx):
        im_path = os.path.join(self.img_dir, self.img_files[idx])
        im = Image.open(im_path)
        
        mask_path = os.path.join(self.mask_dir, self.find_mask(self.img_files[idx]))
        mask = Image.open(mask_path)
        
        if self.transform:
            im = self.transform(im)
            mask = transforms.ToTensor()(mask)
            mask = transforms.Resize((256,256))(mask)
            
        return im, mask
    
'''
    calculate mean and standard deviation
    modified from Binary Study - How to Normalize tutorial
'''
def calc_mean_std(loader):
    cnt = 0
    mom1 = torch.empty(3)
    mom2 = torch.empty(3)

    for im, _ in loader:
        b, c, h, w = im.shape
        num_pix = b * h * w
        
        sums = torch.sum(im, axis=[0,2,3])
        sumsq = torch.sum(im**2, axis=[0,2,3])
        
        mom1 = (cnt * mom1 + sums)/(cnt + num_pix)
        mom2 = (cnt * mom2 +sumsq)/(cnt + num_pix)
        cnt += num_pix

    mean = mom1
    std = torch.sqrt(mom2 - mom1**2)

    return mean, std