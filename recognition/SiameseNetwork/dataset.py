# containing the data loader for loading and preprocessing your data

import os
import random
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

device = torch.device('cuda')

class CustomDataset(Dataset):
    def __init__(self, ad_dir, nc_dir, transform=None, validate=False, split_ratio=0.8):
        # get the file path
        self.ad_folder = ad_dir
        self.nc_folder = nc_dir

        # get the samples' name
        self.ad_names = os.listdir(ad_dir)
        self.nc_names = os.listdir(nc_dir)

        # define the transform
        self.transform = transform

        # splite data to train set and validation set
        total_ad_samples = len(self.ad_names)
        split_ad_samples = int(total_ad_samples * split_ratio)
        total_nc_samples = len(self.nc_names)
        split_nc_samples = int(total_nc_samples * split_ratio)

        if validate:
            self.ad_names = self.ad_names[split_ad_samples:]
            self.nc_names = self.nc_names[split_nc_samples:]
        else:
            self.ad_names = self.ad_names[:split_ad_samples]
            self.nc_names = self.nc_names[:split_nc_samples]


    def __len__(self):
        return 2 * min(len(self.ad_names), len(self.nc_names))

    def __getitem__(self, index):

        # path to the image, randomly select ad image and nc image
        # chooce anchor image evently from ad image set and nc image set
        if index % 2 == 0:
            anchor_path = os.path.join(self.ad_folder, self.ad_names[index//2])
            positive_path = os.path.join(self.ad_folder, random.choice(self.ad_names))
            negative_path = os.path.join(self.nc_folder, random.choice(self.nc_names))
        else:
            anchor_path = os.path.join(self.nc_folder, self.nc_names[index//2])
            positive_path = os.path.join(self.nc_folder, random.choice(self.nc_names))
            negative_path = os.path.join(self.ad_folder, random.choice(self.ad_names))

        # open images
        anchor = Image.open(anchor_path)
        positive = Image.open(positive_path)
        negative = Image.open(negative_path)

        # apply transformation
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        
        return anchor, positive, negative

# calculate the mean and std of the dataset
# input: The folder containing folders containing images
# outupt: mean and std of all images across all subfolders
def compute_mean_std(img_folder):
    # get subfolders
    subfolders = [dir for dir in os.listdir(img_folder) if os.path.isdir(os.path.join(img_folder, dir))]

    # transformer
    transform = transforms.Compose([
        # transform image from numpy.ndarray to tensor
        # and normalize pixels from 0~255 to 0~1
        transforms.ToTensor()
        ])

    num_px = 0
    sum_px = 0
    sum_px_sq = 0

    for subfolder in subfolders:
        subfolder_path = os.path.join(img_folder, subfolder)
        img_names = os.listdir(subfolder_path)

        for img_name in img_names:
            # open the image and put them into GPU
            img_path = os.path.join(subfolder_path, img_name)
            img = Image.open(img_path)
            img_tensor = transform(img).to(device)

            num_px += img_tensor.numel()  # get the # of px
            sum_px += torch.sum(img_tensor)
            sum_px_sq += torch.sum(img_tensor ** 2)

    # calculate mean and std for all images across all subfolders
    mean = sum_px / num_px
    std = torch.sqrt((sum_px_sq / num_px) - (mean ** 2))

    return mean.item(), std.item()

# a function to load all required data
def load_data(train_folder_path, train_ad_path, train_nc_path, test_ad_path, test_nc_path, batch_size=32):
    # calculate mean and std for train set
    mean, std = compute_mean_std(train_folder_path)

    # define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])

    # create dataset
    train_set = CustomDataset(ad_dir=train_ad_path, nc_dir=train_nc_path, transform=transform, validate=False, split_ratio=0.8)
    validation_set = CustomDataset(ad_dir=train_ad_path, nc_dir=train_nc_path, transform=transform, validate=True, split_ratio=0.8)
    test_set = CustomDataset(ad_dir=test_ad_path, nc_dir=test_nc_path, transform=transform, validate=False, split_ratio=1)

    # create dataloader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, validation_loader, test_loader