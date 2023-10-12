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
# TESTIMAGEPATH = '../../groups/comp3710/ADNI/AD_NC/test'
# TRAINIMAGEPATH = '../../groups/comp3710/ADNI/AD_NC/train'

# # Local Locations
# TESTIMAGEPATH = '../ADNI/AD_NC/test'
# TRAINIMAGEPATH = '../ADNI/AD_NC/train'

# Local Locations
TESTIMAGEPATH = '..\\ADNI\\AD_NC\\test'
TRAINIMAGEPATH = '..\\ADNI\\AD_NC\\train'

# Creating Lists of Directories
train_dirs_full = []
train_dirs_AD = os.listdir(TRAINIMAGEPATH + '\\AD')
train_dirs_NC = os.listdir(TRAINIMAGEPATH + '\\NC')

test_dirs_full = []
test_dirs_AD = os.listdir(TESTIMAGEPATH + '\\AD')
test_dirs_NC = os.listdir(TESTIMAGEPATH + '\\NC')

train_dirs_full_brain = []
test_dirs_full_brain = []

# Appending full image paths.
for i in os.listdir(TRAINIMAGEPATH):
    if i == 'AD':
        for j in train_dirs_AD:
            train_dirs_full.append(os.path.join(TRAINIMAGEPATH, 'AD', j))
    else:
        for j in train_dirs_NC:
            train_dirs_full.append(os.path.join(TRAINIMAGEPATH, 'NC', j))

for i in os.listdir(TESTIMAGEPATH):
    if i == 'AD':
        for j in test_dirs_AD:
            test_dirs_full.append(os.path.join(TESTIMAGEPATH, 'AD', j))
    else:
        for j in test_dirs_NC:
            test_dirs_full.append(os.path.join(TESTIMAGEPATH, 'NC', j))

train_brain_sets_AD = len(train_dirs_AD)/20
train_brain_sets_NC = len(train_dirs_NC)/20

test_brain_sets_AD = len(test_dirs_AD)/20
test_brain_sets_NC = len(test_dirs_NC)/20

# Appending full brain paths.
for i in os.listdir(TRAINIMAGEPATH):
    if i == 'AD':
        for s in range(int(train_brain_sets_AD)):
            temp_full_paths = []
            for j in train_dirs_AD[20*s:20*s + 20]:
                temp_full_paths.append(os.path.join(TRAINIMAGEPATH, 'AD', j))
            train_dirs_full_brain.append(temp_full_paths)
    else:
        for s in range(int(train_brain_sets_NC)):
            temp_full_paths = []
            for j in train_dirs_NC[20*s:20*s + 20]:
                temp_full_paths.append(os.path.join(TRAINIMAGEPATH, 'NC', j))
            train_dirs_full_brain.append(temp_full_paths)

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

#print(train_dirs_full)

##### Temp Code for Visualising Images.
# fig, axs = plt.subplots(2)
# image1 = train_dirs_full[1]
#
# data = cv2.imread(image1)
# print(data.shape)
# axs[0].imshow(data)
# data2 = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
# axs[1].imshow(data2)
#
# plt.show()

image_size = 32
# Only works if 32? do I need to change resnet?

# Transforms to be applied to data loaders.
transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.ToTensor(),
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
        self.label_list = [image.split('\\')[-2] for image in self.images]

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        # Read image and extract anchor label.
        anchor_data = cv2.imread(img_path)
        anchor_label = img_path.split('\\')[-2]

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

        # fig, axs = plt.subplots(3)
        #
        # axs[0].imshow(anchor_data)
        # axs[1].imshow(positive_image)
        # axs[2].imshow(negative_image)
        #
        # plt.show()

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

    def __len__(self):
        return len(self.image_segs)

    def __getitem__(self, idx):
        img_path = self.image_segs[idx]
        # Read image and extract anchor label.

        anchor_data = []
        positive_image = []
        negative_image = []

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

        # For each slice of the brain.
        for i in range(20):
            anchor_data.append(cv2.imread(img_path[i]))

            # If training, use triplet.
            if not self.clas:

                # Convert paths into images.

                positive_image.append(cv2.imread(positive[i]))
                negative_image.append(cv2.imread(negative[i]))

            # # If transforms are given.
            # if self.transform is not None:
            #     # apply the transformations to image
            #     # e.g. transform [H, W, n] format to [n, H, W]
            #     anchor_data = self.transform(anchor_data)
            #     if self.clas:
            #         positive_image = self.transform(positive_image)
            #         negative_image = self.transform(negative_image)

        if anchor_label == 'AD':
            anchor_label = 1
        else:
            anchor_label = 0

        test1 = positive[0].split('\\')[-2]
        test2 = negative[0].split('\\')[-2]

        if not self.clas:
            return anchor_data, positive_image, negative_image, anchor_label
        else:
            return anchor_data, anchor_label


def get_dataset(train, clas):
    # if train == 1 and clas == 0:
    #     return ImageDataset(train_dirs_full, transform=transform, train=1)
    # elif train == 1 and clas == 1:
    #     return ImageDataset(train_dirs_full, transform=transform, train=0)
    # else:
    #     return ImageDataset(test_dirs_full, transform=transform, train=0)

    if train == 1 and clas == 0:
        return ImageDataset3D(train_dirs_full_brain, transform=transform, clas=0)
    elif train == 1 and clas == 1:
        return ImageDataset3D(train_dirs_full_brain, transform=transform, clas=1)
    else:
        return ImageDataset3D(test_dirs_full_brain, transform=transform, clas=0)

