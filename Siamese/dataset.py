import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class ADNIDataset(Dataset):
    def __init__(self, data_path, transform=None):
        super(ADNIDataset, self).__init__()

        self.transform = transform

        self.ad_path = os.path.join(data_path, 'AD')
        self.nc_path = os.path.join(data_path, 'NC')

        self.ad_images = os.listdir(self.ad_path)
        self.nc_images = os.listdir(self.nc_path)

    def __len__(self):
        return min(len(self.ad_images), len(self.nc_images))

    def load_image(self, path, image_name):
        img = Image.open(os.path.join(path, image_name)).convert('L')
        if self.transform:
            img = self.transform(img)
        return img

    def __getitem__(self, index):
        if index % 2 == 0:
            # Positive example (both images are AD)
            img1 = self.load_image(self.ad_path, self.ad_images[index % len(self.ad_images)])
            img2 = self.load_image(self.ad_path, self.ad_images[(index + 1) % len(self.ad_images)])
            label = torch.tensor(1, dtype=torch.float)
        else:
            # Negative example (one image is AD, the other is NC)
            img1 = self.load_image(self.ad_path, self.ad_images[index % len(self.ad_images)])
            img2 = self.load_image(self.nc_path, self.nc_images[index % len(self.nc_images)])
            label = torch.tensor(0, dtype=torch.float)

        return img1, img2, label

