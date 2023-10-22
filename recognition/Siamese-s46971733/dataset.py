# Imports
import time
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2
import os
import random

# Rangpur Locations
TESTIMAGEPATH = '../../groups/comp3710/ADNI/AD_NC/test'
TRAINIMAGEPATH = '../../groups/comp3710/ADNI/AD_NC/train'

# Local Locations
#TESTIMAGEPATH = '../ADNI/AD_NC/test'
#TRAINIMAGEPATH = '../ADNI/AD_NC/train'
#TRAINIMAGEPATH = '../ADNI/AD_NC/train_big'

# Creating Lists of Directories
train_dirs_AD = sorted(os.listdir(TRAINIMAGEPATH + '/AD'))
train_dirs_NC = sorted(os.listdir(TRAINIMAGEPATH + '/NC'))

test_dirs_AD = sorted(os.listdir(TESTIMAGEPATH + '/AD'))
test_dirs_NC = sorted(os.listdir(TESTIMAGEPATH + '/NC'))

# Storage for Full Paths.
train_dirs_full = []
test_dirs_full = []
valid_dirs_full = []

temp_NC_full = []
temp_AD_full = []

# For 3D Version.
train_dirs_full_brain = []
test_dirs_full_brain = []
valid_dirs_full_brain = []

temp_NC_full_brain = []
temp_AD_full_brain = []

# Appending full image paths.
for i in os.listdir(TRAINIMAGEPATH):
    if i == 'AD':
        for j in train_dirs_AD:
            temp_AD_full.append(os.path.join(TRAINIMAGEPATH, 'AD', j))
        # Separate train dirs into train and valid.
        train_len = int(round(0.8*len(temp_AD_full)/20))
        train_dirs_full = temp_AD_full[:20*train_len]
        valid_dirs_full = temp_AD_full[20*train_len:]

    else:
        for j in train_dirs_NC:
            temp_NC_full.append(os.path.join(TRAINIMAGEPATH, 'NC', j))
        # Separate train dirs into train and valid.
        train_len = int(round(0.8 * len(temp_NC_full) / 20))
        train_dirs_full_NC = temp_NC_full[:20 * train_len]
        valid_dirs_full_NC = temp_NC_full[20 * train_len:]

        # Extend Train and Valid directories with the NC directories.
        train_dirs_full.extend(train_dirs_full_NC)
        valid_dirs_full.extend(valid_dirs_full_NC)

for i in os.listdir(TESTIMAGEPATH):
    if i == 'AD':
        for j in test_dirs_AD:
            test_dirs_full.append(os.path.join(TESTIMAGEPATH, 'AD', j))
    else:
        for j in test_dirs_NC:
            test_dirs_full.append(os.path.join(TESTIMAGEPATH, 'NC', j))

#############################################################
# Creating Lists of Full Brains (20 Slices) for 3D Dataset. #
#############################################################

train_brain_sets_AD = len(train_dirs_AD)/20
train_brain_sets_NC = len(train_dirs_NC)/20

test_brain_sets_AD = len(test_dirs_AD)/20
test_brain_sets_NC = len(test_dirs_NC)/20

# Appending full brain paths.

for s in range(int(train_brain_sets_AD)):
    temp_full_paths = []
    for j in train_dirs_AD[20*s:20*s + 20]:
        temp_full_paths.append(os.path.join(TRAINIMAGEPATH, 'AD', j))
    temp_AD_full_brain.append(temp_full_paths)
# Separate train dirs into train and valid.
train_len = int(round(0.8 * len(temp_AD_full_brain)))
train_dirs_full_brain = temp_AD_full_brain[:train_len]
valid_dirs_full_brain = temp_AD_full_brain[train_len:]

for s in range(int(train_brain_sets_NC)):
    temp_full_paths = []
    for j in train_dirs_NC[20*s:20*s + 20]:
        temp_full_paths.append(os.path.join(TRAINIMAGEPATH, 'NC', j))
    temp_NC_full_brain.append(temp_full_paths)
# Separate train dirs into train and valid.
train_len = int(round(0.8 * len(temp_NC_full_brain)))
train_dirs_full_NC_brain = temp_NC_full_brain[:train_len]
valid_dirs_full_NC_brain = temp_NC_full_brain[train_len:]

# Extend Train and Valid directories with the NC directories.
train_dirs_full_brain.extend(train_dirs_full_NC_brain)
valid_dirs_full_brain.extend(valid_dirs_full_NC_brain)

for i in os.listdir(TESTIMAGEPATH):
    if i == 'AD':
        for s in range(int(test_brain_sets_AD)):
            temp_full_paths = []
            for j in test_dirs_AD[20*s:20*s + 20]:
                temp_full_paths.append(os.path.join(TESTIMAGEPATH, 'AD', j))
            test_dirs_full_brain.append(temp_full_paths)
    else:
        for s in range(int(test_brain_sets_NC)):
            temp_full_paths = []
            for j in test_dirs_NC[20*s:20*s + 20]:
                temp_full_paths.append(os.path.join(TESTIMAGEPATH, 'NC', j))
            test_dirs_full_brain.append(temp_full_paths)

image_size = 210

random.shuffle(train_dirs_full_brain)

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((image_size, image_size), antialias=None)
                                 ])

class ImageDataset(Dataset):
    """
    Custom Dataset made by splitting ADNI paths into anchors, positives, negatives and labels.

    __getitem__ method based on https://medium.com/@Skpd/triplet-loss-on-imagenet-dataset-a2b29b8c2952
      - Modified to fit dataset and for use in train.py
    """
    def __init__(self, image_dir, transform=None, train=0):
        self.images = image_dir
        self.train = train

        # Create list of image labels.
        self.label_list = [image.split('/')[-2] for image in self.images]

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        # Read image and extract anchor label.
        anchor_data = cv2.imread(img_path)
        anchor_label = img_path.split('/')[-2]

        # If training, use triplet.
        if self.train:
            positive_list = []
            negative_list = []

            # Obtain list of positive and negative images.
            for i in range(len(self.images)):
                if self.images[i] == img_path:
                    continue
                elif self.label_list[i] == anchor_label:
                    positive_list.append(self.images[i])
                else:
                    negative_list.append(self.images[i])

            # Randomly select a positive and negative image from list.
            positive = random.choice(positive_list)
            negative = random.choice(negative_list)

            # Convert paths into images.
            positive_image = cv2.imread(positive)
            negative_image = cv2.imread(negative)

        # If transforms are given.
        if self.transform is not None:
            # apply the transformations to image
            # e.g. transform [H, W, n] format to [n, H, W]
            anchor_data = self.transform(anchor_data)
            if self.train:
                positive_image = self.transform(positive_image)
                negative_image = self.transform(negative_image)

        if anchor_label == 'AD':
            anchor_label = 1
        else:
            anchor_label = 0

        if self.train:
            return anchor_data, positive_image, negative_image, anchor_label
        else:
            return anchor_data, anchor_label

class ImageDataset3D(Dataset):
    """
    Custom Dataset made by splitting ADNI paths into anchors, positives, negatives and labels.

    __getitem__ method based on https://medium.com/@Skpd/triplet-loss-on-imagenet-dataset-a2b29b8c2952
      - Modified to fit dataset and for use in train.py
    """
    def __init__(self, image_dir, transform=None, clas=0):
        self.image_segs = image_dir
        self.clas = clas

        # Create list of image labels.
        self.label_list = [image[0].split('\\')[-2] for image in self.image_segs]
        self.transform = transform

    # Create transforms randomly to be applied to all 20 slices of a brain.
    def create_transform(self):
        random_h = random.randint(0, 1)
        random_v = random.randint(0, 1)

        if random_h and not random_v:
            transform_train = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Resize((image_size, image_size), antialias=None),
                                                  transforms.RandomHorizontalFlip(p=1)
                                                  ])

        elif random_v and not random_h:
            transform_train = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Resize((image_size, image_size), antialias=None),
                                                  transforms.RandomVerticalFlip(p=1)
                                                  ])

        elif random_h and random_v:
            transform_train = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Resize((image_size, image_size), antialias=None),
                                                  transforms.RandomHorizontalFlip(p=1),
                                                  transforms.RandomVerticalFlip(p=1)
                                                  ])

        else:
            transform_train = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Resize((image_size, image_size), antialias=None)
                                                  ])

        return transform_train

    def __len__(self):
        return len(self.image_segs)

    def __getitem__(self, idx):

        # Read image and extract anchor label.
        img_path = self.image_segs[idx]
        anchor_full_data = []
        positive_images = []
        negative_images = []

        # Split path to get label.
        anchor_label = img_path[0].split('\\')[-2]

        if not self.clas:
            positive_list = []
            negative_list = []

            # Obtain list of positive and negative images.
            for j in range(len(self.image_segs)):
                if self.image_segs[j] == img_path:
                    continue
                elif self.label_list[j] == anchor_label:
                    positive_list.append(self.image_segs[j])
                else:
                    negative_list.append(self.image_segs[j])

            # Randomly select a positive and negative image from list.
            positive = random.choice(positive_list)
            negative = random.choice(negative_list)

        anchortransform = self.create_transform()
        postransform = self.create_transform()
        negtransform = self.create_transform()

        # For each slice of the brain.
        for i in range(20):

            anchor_data = cv2.imread(img_path[i], cv2.COLOR_BGR2GRAY)

            # If transforms are given.
            if self.transform is not None:
                # Apply the transformations to image
                anchor_data = anchortransform(anchor_data)

            anchor_full_data.append(anchor_data)

            # If training, use triplet.
            if not self.clas:

                # Convert paths into images.
                positive_image = cv2.imread(positive[i], cv2.COLOR_BGR2GRAY)
                negative_image = cv2.imread(negative[i], cv2.COLOR_BGR2GRAY)

                # If transforms are given.
                if self.transform is not None:
                    positive_image = postransform(positive_image)
                    negative_image = negtransform(negative_image)

                positive_images.append(positive_image)
                negative_images.append(negative_image)

        if anchor_label == 'AD':
            anchor_label = 1
        else:
            anchor_label = 0

        anchor_full_data = torch.stack(anchor_full_data, dim=1)

        if not self.clas:
            positive_images = torch.stack(positive_images, dim=1)
            negative_images = torch.stack(negative_images, dim=1)
            return anchor_full_data, positive_images, negative_images, anchor_label
        else:
            return anchor_full_data, anchor_label

# Used in train.py to create datasets for data loader.
def get_dataset(train=0, clas=0, valid=0):
    if train == 1 and clas == 0:
        return ImageDataset3D(train_dirs_full_brain, transform=transform, clas=0)
    elif train == 1 and clas == 1:
        return ImageDataset3D(train_dirs_full_brain, transform=transform, clas=1)
    elif valid == 1 and clas == 0:
        return ImageDataset3D(valid_dirs_full_brain, transform=transform, clas=0)
    elif valid == 1 and clas == 1:
        return ImageDataset3D(valid_dirs_full_brain, transform=transform, clas=1)
    else:
        return ImageDataset3D(test_dirs_full_brain, transform=transform, clas=1)
