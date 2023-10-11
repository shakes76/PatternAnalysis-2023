import glob
import torch
import nibabel
import numpy as np
import torchio as tio
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class NiiImageLoader(DataLoader):
    def __init__(self, image_path, mask_path):
        self.inputs = []
        self.masks = []
        # retrieve path from dataset
        for f in sorted(glob.iglob(image_path)):
            self.inputs.append(f)
        for f in sorted(glob.iglob(mask_path)):
            self.masks.append(f)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.inputs)

    # open files
    def __getitem__(self, idx):
        image_p = self.inputs[idx]
        mask_p = self.masks[idx]

        image = nibabel.load(image_p)
        image = np.asarray(image.dataobj)

        mask = nibabel.load(mask_p)
        mask = np.asarray(mask.dataobj)

        image = self.to_tensor(image)
        image = image.unsqueeze(0)
        image = image.data

        mask = self.to_tensor(mask)
        mask = mask.unsqueeze(0)
        mask = mask.data

        return image, mask


class Augment:
    """Class for data augmentation"""
    def __init__(self):
        self.shrink = tio.CropOrPad((16, 32, 32))
        self.flip0 = tio.transforms.RandomFlip(0, flip_probability=1)  # flip the data randomly
        self.flip1 = tio.transforms.RandomFlip(1, flip_probability=1)
        self.flip2 = tio.transforms.RandomFlip(2, flip_probability=1)

        nothing = tio.transforms.RandomFlip(0, flip_probability=0)
        bias_field = tio.transforms.RandomBiasField()
        blur = tio.transforms.RandomBlur()
        spike = tio.transforms.RandomSpike()
        prob = {nothing: 0.7, bias_field: 0.1, blur: 0.1, spike: 0.1}
        self.oneof = tio.transforms.OneOf(prob)  # randomly choose one augment method from the three

    def crop_and_augment(self, image, mask):
        image = self.shrink(image)
        mask = self.shrink(mask)
        image = self.oneof(image)

        return image, mask


# # load the dataset
dataset = NiiImageLoader("/home/groups/comp3710/HipMRI_Study_open/semantic_MRs/*",
                         "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only/*")

# split the dataset
trainloader, valloader, testloader = torch.utils.data.random_split(dataset, [179, 16, 16])
