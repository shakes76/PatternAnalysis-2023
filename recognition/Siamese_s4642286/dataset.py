"""
Name: dataset.py
Student: Ethan Pinto (s4642286)
Description: Creates the data loader for loading and preprocessing the ADNI Brain Data.
"""

import numpy as np
import itertools
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image


# dataroot = "/home/groups/comp3710"
train_dataroot = "C:/Users/Q/OneDrive/Desktop/COMP3710/REPORT/ADNI/AD_NC/train"
test_dataroot = "C:/Users/Q/OneDrive/Desktop/COMP3710/REPORT/ADNI/AD_NC/test"


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to a common size
    transforms.ToTensor()  # Convert images to tensors
])


train_dataset = datasets.ImageFolder(root=train_dataroot, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dataroot, transform=transform)

# Assuming AD is class 0 and NC is class 1
train_dataset.class_to_idx = {'AD': 0, 'NC': 1}
test_dataset.class_to_idx = {'AD': 0, 'NC': 1}
 

def make_paired_datasets(dataset):
    X_pairs, y_pairs = [], []

    for t in itertools.product(dataset, dataset):
        pair_A, pair_B = t
        img_A, label_A = t[0]
        img_B, label_B = t[1]

        new_label = int(label_A == label_B)

        X_pairs.append([img_A, img_B])
        y_pairs.append(new_label)

    X_pairs = np.array(X_pairs)
    y_pairs = np.array(y_pairs)

    return X_pairs, y_pairs


X_train, y_train = make_paired_datasets(train_dataset.samples)
X_test, y_test = make_paired_datasets(test_dataset.samples)

train_set = TensorDataset(X_train, y_train)    # Wrap X and Y into a single training dataset
test_set = TensorDataset(X_test, y_test)       # Wrap X and Y into a single test dataset

# Define the data loaders
trainloader = DataLoader(train_set, batch_size=64, shuffle=True)
testloader = DataLoader(test_set, batch_size=64, shuffle=False)


# HARDCODED VALIDATION SET FOR MODEL EVALUATION WHILE TRAINING

# TEST SET DOES NOT HAVE TO BE PAIRS!! since the model now only takes in one image.