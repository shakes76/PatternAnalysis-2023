from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np

class customDataset(Dataset):
    def __init__(self, data_path, GT_path, d_transform, g_transform):
        super(customDataset, self).__init__()

        self.f_path = data_path
        self.image_path = sorted(os.listdir(data_path))
        
        #self.image_path.remove('LICENCE.txt')
        #self.image_path.remove('ATTRIBUTION.txt')

        self.g_path = GT_path
        self.GT_path = sorted(os.listdir(GT_path))
        #self.GT_path.remove('LICENCE.txt')
        #self.GT_path.remove('ATTRIBUTION.txt')

        self.data_transform = d_transform
        self.gt_transform = g_transform
        pass

    def __len__(self):
        if len(self.image_path) == len(self.GT_path):
            return(len(self.image_path))
        else:
            print(">>data and ground truth not same lenght")

    def __getitem__(self, idx):    
        img_name = os.path.join(self.f_path, self.image_path[idx])
        img = Image.open(img_name)

        GT_name = os.path.join(self.g_path, self.GT_path[idx])
        GT = Image.open(GT_name)

        if self.data_transform:
            img = self.data_transform(img)
            GT = self.gt_transform(GT)


        return img, GT

        
