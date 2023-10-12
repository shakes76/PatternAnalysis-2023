import torch
import os
from PIL import Image
from torch.utils.data import Dataset



class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super(CustomDataset,self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(root_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)

        return image
