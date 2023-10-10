# containing the source code for training, validating, testing and saving your model. 
# The model should be imported from “modules.py” and the data loader should be imported from “dataset.py”. 
# Make sure to plot the losses and metrics during training

import dataset, modules
import torch
import torchvision.transforms as transforms 

train_folder_path = "D:/Study/MLDataSet/AD_NC/train"
AD_path = "D:/Study/MLDataSet/AD_NC/train/AD"
NC_path = "D:/Study/MLDataSet/AD_NC/train/NC"

# calculate mean and std for train set
mean, std = dataset.compute_mean_std(train_folder_path)

# define transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
    ])

# train_set = dataset.CustomDataset(ad_dir=AD_path, nc_dir=NC_path, transform=transform)
