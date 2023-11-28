import glob
import torch
import nibabel
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


# Dataloader for .nii.gz data
class NiiImageLoader(DataLoader):
    def __init__(self, image_path, mask_path):
        self.inputs = []
        self.masks = []
        # retrieve path from dataset
        for input in sorted(glob.iglob(image_path)):
            self.inputs.append(input)
        for mask in sorted(glob.iglob(mask_path)):
            self.masks.append(mask)

    def __len__(self):
        return len(self.inputs)

    # load files
    def __getitem__(self, idx):
        image_p = self.inputs[idx]
        mask_p = self.masks[idx]

        # load nii image with nibabel
        image = nibabel.load(image_p)
        # transform as numpy array
        image = np.asarray(image.dataobj)
        # transform as tensor
        image = transforms.ToTensor(image).unsqueeze(0).data

        mask = nibabel.load(mask_p)
        mask = np.asarray(mask.dataobj)
        mask = transforms.ToTensor(mask).unsqueeze(0).data

        return image, mask


# load the dataset in rangpur
dataset = NiiImageLoader("/home/groups/comp3710/HipMRI_Study_open/semantic_MRs/*",
                         "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only/*")


# split the dataset as 85% training, 7.5% validation, 7.5% testing
trainloader, valloader, testloader = torch.utils.data.random_split(dataset, [179, 16, 16])
