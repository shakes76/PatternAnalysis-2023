'''
Data loader for loading and preprocessing ADNI data.
'''
import os
from typing import Any, Callable, Optional, Tuple

import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Dataset

# Root directory for ADNI training and testing split
ADNI_ROOT = os.path.join('ADNI_AD_NC_2D', 'AD_NC')

class ADNI(Dataset):
    def __init__(self, root: str, train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.img_dir = os.path.join(root, 'train' if train else 'test')
        # Read and store file names of all images for faster access
        ad_fnames = os.listdir(os.path.join(self.img_dir, 'AD'))
        nc_fnames = os.listdir(os.path.join(self.img_dir, 'NC'))
        self.img_list = ad_fnames + nc_fnames
        # Pre-calculate and store number of images in either class
        self.count_ad = len(ad_fnames)
        self.count = self.count_ad + len(nc_fnames)

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        subdir = 'AD' if index < self.count_ad else 'NC'
        img_path = os.path.join(self.img_dir, subdir, self.img_list[index])
        image = torchvision.io.read_image(img_path, torchvision.io.ImageReadMode.RGB)
        label = int(index < self.count_ad) # return 1 for AD and 0 for NC
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def create_train_dataloader() -> DataLoader:
    '''
    Returns a DataLoader on pre-processed training data from the ADNI dataset.
    '''
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ConvertImageDtype(torch.float),
    ])
    train_dataset = ADNI(ADNI_ROOT, train=True, transform=transform)
    return DataLoader(train_dataset, batch_size=4, shuffle=True)

def create_test_dataloader() -> DataLoader:
    '''
    Returns a DataLoader on pre-processed test data from the ADNI dataset.'''
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ConvertImageDtype(torch.float),
    ])
    test_dataset = ADNI(ADNI_ROOT, train=False, transform=transform)
    return DataLoader(test_dataset, batch_size=64, shuffle=True)
