'''
Data loader for loading and preprocessing ADNI data.
'''
import copy
import os
import random
import sys

from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Dataset

# Root directory for ADNI training and testing split
ADNI_ROOT = {
    'win32': Path('ADNI', 'AD_NC'),
    'linux': Path('/home', 'groups', 'comp3710', 'ADNI', 'AD_NC')
}[sys.platform]

class ADNI(Dataset):
    def __init__(self, root: str, train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        self.transform = transform
        self.target_transform = target_transform
        # Read and store file names of all images for faster access
        self.img_dir = Path(root, 'train' if train else 'test')
        self.ad_fnames = os.listdir(Path(self.img_dir, 'AD'))
        self.nc_fnames = os.listdir(Path(self.img_dir, 'NC'))
        # Pre-calculate and store number of images in either class
        self.count_ad = len(self.ad_fnames)
        self.count_nc = len(self.nc_fnames)
        self.count = self.count_ad + self.count_nc

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # Fetch image and determine label
        if index < self.count_ad:
            label = 1
            path = Path(self.img_dir, 'AD', self.ad_fnames[index])
        else:
            label = 0
            path = Path(self.img_dir, 'NC', self.nc_fnames[index - self.count_ad])
        image = torchvision.io.read_image(str(path), torchvision.io.ImageReadMode.RGB)
        # Apply image and labels transforms (if any specified)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def train_val_split(dataset: ADNI, val_pct: float) -> Tuple[Dataset, Dataset]:
    '''Split dataset into random train and validation subsets.'''

    def patient_level_split(fnames: List) -> Tuple[List, List]:
        '''Patient-level split array into random train and validation subsets.'''
        nonlocal val_pct
        pids = set(fname.split('_')[0] for fname in fnames)
        pids_valid = set(random.sample(list(pids), int(len(pids) * val_pct)))
        fnames_train = []; fnames_valid = []
        for fname in fnames:
            if fname.split('_')[0] in pids_valid:
                fnames_valid.append(fname)
            else:
                fnames_train.append(fname)
        return fnames_train, fnames_valid

    # Deepcopy one dataset so training and validation sets are unlinked
    valid_dataset = copy.deepcopy(dataset)

    # Apply patient-level split to AD and NC training samples separately
    train_ad, valid_ad = patient_level_split(copy.deepcopy(dataset.ad_fnames))
    train_nc, valid_nc = patient_level_split(copy.deepcopy(dataset.nc_fnames))

    # Update training dataset with new splits
    dataset.ad_fnames = train_ad
    dataset.nc_fnames = train_nc
    dataset.count_ad = len(train_ad)
    dataset.count_nc = len(train_nc)
    dataset.count = dataset.count_ad + dataset.count_nc

    # Update validation dataset with new splits
    valid_dataset.ad_fnames = valid_ad
    valid_dataset.nc_fnames = valid_nc
    valid_dataset.count_ad = len(valid_ad)
    valid_dataset.count_nc = len(valid_nc)
    valid_dataset.count = valid_dataset.count_ad + valid_dataset.count_nc

    return dataset, valid_dataset

def create_train_dataloader(val_pct: float = 0.2) -> DataLoader:
    '''
    Returns a DataLoader on pre-processed training data from the ADNI dataset.
    '''
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ConvertImageDtype(torch.float),
    ])
    train_dataset = ADNI(ADNI_ROOT, train=True, transform=transform)
    if val_pct == 0:
        return DataLoader(train_dataset, batch_size=64, shuffle=True)
    else:
        train_dataset, valid_dataset = train_val_split(train_dataset, val_pct)
        return (DataLoader(train_dataset, batch_size=64, shuffle=True),
                DataLoader(valid_dataset, batch_size=64, shuffle=True))

def create_test_dataloader() -> DataLoader:
    '''
    Returns a DataLoader on pre-processed test data from the ADNI dataset.'''
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ConvertImageDtype(torch.float),
    ])
    test_dataset = ADNI(ADNI_ROOT, train=False, transform=transform)
    return DataLoader(test_dataset, batch_size=64, shuffle=True)
