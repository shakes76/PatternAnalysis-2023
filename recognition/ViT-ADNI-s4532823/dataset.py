import torch
from torch.utils.data import DataLoader, Dataset
import os
from pathlib import Path
import sys
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import random
import math
import copy

# ADNI data path on Rangpur
ADNI_PATH = Path('/home', 'groups', 'comp3710', 'ADNI', 'AD_NC')
# ADNI data path on personal computer
if sys.platform == 'win32':
    ADNI_PATH = Path('D:', 'ADNI', 'AD_NC')

class ADNIDataset(Dataset):
    def __init__(self, root_path, train=True, transform=None):
        self.path = Path(root_path, "train" if train else 'test')
        self.transform = transform
        self.ad_files = os.listdir(Path(self.path, 'AD'))
        self.nc_files = os.listdir(Path(self.path, 'NC'))

    def __len__(self):
        return len(self.ad_files) + len(self.nc_files)

    def __getitem__(self, idx):
        assert idx >= 0 and idx < self.__len__(), "Index out of range."
        # Index in AD range
        if idx < len(self.ad_files):
            img_filename = self.ad_files[idx]
            image = Image.open(Path(self.path, 'AD', img_filename))
            label = "AD"
        # Index in NC range
        else:
            img_filename = self.nc_files[idx-len(self.ad_files)]
            image = Image.open(Path(self.path, 'NC', img_filename))
            label = "NC"
        # Apply transform if present
        if self.transform:
            image = self.transform(image)
        return image, label
    
"""
Function to split the training set into training and validation sets, with a certain proportion of images to be included in the validation set.

Images to be split on a patient level (i.e. all images with the same patient ID to be included in the same set).
"""
def split_train_val(dataset: ADNIDataset, val_proportion: float):
    # Get the patient ID's and the number of AD and NC patients
    ad_patient_ids = set(filename.split('_')[0] for filename in dataset.ad_files)
    num_ad_patients = len(ad_patient_ids)
    nc_patient_ids = set(filename.split('_')[0] for filename in dataset.nc_files)
    num_nc_patients = len(nc_patient_ids)

    # Generate a random sample of patient ID's for the validation set
    ad_pids_val = random.sample(ad_patient_ids, math.floor(num_ad_patients*val_proportion))
    nc_pids_val = random.sample(nc_patient_ids, math.floor(num_nc_patients*val_proportion))

    # Make the validation dataset a deep copy of the training dataset
    val_dataset = copy.deepcopy(dataset)

    # Update each dataset's files by patient level split above 
    dataset.ad_files = [file for file in dataset.ad_files if file.split('_')[0] not in ad_pids_val]
    val_dataset.ad_files = [file for file in dataset.ad_files if file.split('_')[0] in ad_pids_val]
    dataset.nc_files = [file for file in dataset.ad_files if file.split('_')[0] not in nc_pids_val]
    val_dataset.nc_files = [file for file in dataset.ad_files if file.split('_')[0] in nc_pids_val]

    return dataset, val_dataset




        

