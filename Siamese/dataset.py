import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ADNIDataset(Dataset):
    def __init__(self, data_path):
        super(ADNIDataset, self).__init__()

        self.transform = transforms.ToTensor()

        self.ad_path = os.path.join(data_path, 'AD')
        self.nc_path = os.path.join(data_path, 'NC')

        self.ad_images = [self.transform(Image.open(os.path.join(self.ad_path, img))) for img in
                          os.listdir(self.ad_path)]
        self.nc_images = [self.transform(Image.open(os.path.join(self.nc_path, img))) for img in
                          os.listdir(self.nc_path)]

        self.ad_images = torch.stack(self.ad_images).unsqueeze(1)
        self.nc_images = torch.stack(self.nc_images).unsqueeze(1)


    def __len__(self):
        return min(len(self.ad_images), len(self.nc_images))

    def __getitem__(self, index):
        if index % 2 == 0:
            # Positive example (both images are AD)
            img1 = self.ad_images[index % len(self.ad_images)]
            img2 = self.ad_images[(index + 1) % len(self.ad_images)]
            label = torch.tensor(1, dtype=torch.float)
        else:
            # Negative example (one image is AD, the other is NC)
            img1 = self.ad_images[index % len(self.ad_images)]
            img2 = self.nc_images[index % len(self.nc_images)]
            label = torch.tensor(0, dtype=torch.float)

        return img1, img2, label


def get_train_dataset(data_path):
    train_dataset = ADNIDataset(os.path.join(data_path, 'train'))
    return train_dataset


def get_test_dataset(data_path):
    test_dataset = ADNIDataset(os.path.join(data_path, 'test'))
    return test_dataset
