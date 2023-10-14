import os
from torch.utils.data import Dataset
from PIL import Image

'''
    custom dataset for ISIC 17/18 Melanoma
'''
class ISICDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.img_files = os.listdir(img_dir)   # TODO only .jpg files
        
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
            mask = self.transform(mask)
            
        return im, mask