import torch
from torch.utils.data import Dataset
import os
from PIL import Image

class customDataset(Dataset):
    def __init__(self, data_path, transform):
        super().__init__()

        self.f_path = data_path
        self.image_path = os.listdir(data_path)
        self.transform = transform
        pass


    def __getitem__(self, idx):
        img_name = os.path.join(self.f_path, self.image_path[idx])
        img = Image.open(img_name)

        if self.transform:
            img = self.transform(img)


    

        
