import os
import numpy as np
import random
from torchvision import datasets, transforms
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import random_split




#-------- TRANSFORMERS FOR TRAINING AND TESTING -----------

def get_transforms_training():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.RandomHorizontalFlip(), # Data Augmenttation
        transforms.ToTensor(),
    ])

def get_transforms_validation():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.ToTensor(),
    ])


def get_transforms_testing():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.ToTensor(),
    ]) 



#--------- CREATE SIAMESE DATASET CLASS --------------------

class SiameseDataset(Dataset):
    def __init__(self, image_list, transforms=None):
        
        self.transforms = transforms
        self.image_list = image_list
        
        self.classes = list(set([label for _, label in image_list]))
        
        # Create pairs and labels
        self.pairs, self.labels = self.make_pairs()

    # function to group the dataset into pairs
    def make_pairs(self):
        # Referred and modified from: https://keras.io/examples/vision/siamese_contrastive/ 

        # Create a dictionary for each class and its corresponding indices
        class_indices = {cls: np.where(np.array([label for _, label in self.image_list]) == cls)[0] 
                 for cls in self.classes}
        
        pairs, labels = [], []
        seen_pairs = set()
        MAX_ATTEMPTS = 100
        
        for idx1, (img1, label1) in enumerate(self.image_list):  

            # Get Positive pair
            idx2 = random.choice(class_indices[label1])
            pair = tuple(sorted([img1, self.image_list[idx2][0]]))
            attempts = 0

            # if chosen pair already created (in seen_pairs) find another partner until there is 
            # a pair that is not in seen_pairs
            while (pair in seen_pairs or idx1 == idx2) and attempts < MAX_ATTEMPTS:
                idx2 = random.choice(class_indices[label1])
                pair = tuple(sorted([img1, self.image_list[idx2][0]]))
                attempts += 1

            # to prevent infinity loop
            if attempts == MAX_ATTEMPTS:
                continue

            seen_pairs.add(pair)
            pairs.append(list(pair))
            # Positive pair is labelled 1
            labels.append(1)

            # Get Negative pair
            label2 = random.choice([cls for cls in self.classes if cls != label1])
            idx2 = random.choice(class_indices[label2])
            pair = tuple(sorted([img1, self.image_list[idx2][0]]))
            attempts = 0
            
             # if chosen pair already created (in seen_pairs) find another partner until there is 
            # a pair that is not in seen_pairs
            while pair in seen_pairs and attempts < MAX_ATTEMPTS:
                idx2 = random.choice(class_indices[label2])
                pair = tuple(sorted([img1, self.image_list[idx2][0]]))
                attempts += 1

            # to prevent infinity loop 
            if attempts == MAX_ATTEMPTS:
                continue
            
            seen_pairs.add(pair)
            pairs.append(list(pair))
             # Negative pair is labelled 0
            labels.append(0)
        
        return pairs, labels

    def __len__(self):
        return len(self.pairs)  # doubled because of pairs

    def __getitem__(self, idx):
        img1_path, img2_path = self.pairs[idx]
        
 
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)

        # Apply transformations, if any
        if self.transforms:
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)

        return img1, img2, torch.tensor(self.labels[idx], dtype=torch.float32), img1_path, img2_path



#------------- FUNCTION TO BE CALLED TO RETURN DATALOADER CONSISTING OF SIAMESE DATASET-----------------


def create_siamese_dataloader(root_dir, batch_size=32, shuffle=True, split_flag=True):
    data = datasets.ImageFolder(root=root_dir, transform=None)
    print("Total Number of images:", len(data))

    # create training and validation pairs to train the siamese network
    if split_flag:
        val_split= 0.2 # validation split
        train_len = int((1.0 - val_split) * len(data))
        val_len = len(data) - train_len

        train_data, val_data = random_split(data, [train_len, val_len])

        train_image_list = [(data.imgs[i][0], data.targets[i]) for i in train_data.indices]
        val_image_list = [(data.imgs[i][0], data.targets[i]) for i in val_data.indices]

        train_siamese_dataset = SiameseDataset(train_image_list, transforms=get_transforms_training())
        val_siamese_dataset = SiameseDataset(val_image_list, transforms=get_transforms_validation())

        print("Training Pairs Length:", len(train_siamese_dataset))
        print("Validation Pairs Length:", len(val_siamese_dataset))

        train_loader = DataLoader(train_siamese_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_siamese_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    else:
        siamese_dataset = SiameseDataset(data.imgs, transforms=get_transforms_testing())
        data_loader = DataLoader(siamese_dataset, batch_size=batch_size, shuffle=False)
        return data_loader


#------------- FUNCTION TO BE CALLED TO RETURN DATALOADER CONSISTING OF CLASSIFIER DATASET-----------------

def get_classification_dataloader(root_dir, batch_size=32, shuffle=True, split_flag=True):
    data = datasets.ImageFolder(root=root_dir, transform=None)
    print("Total Number of images:", len(data))
    print(data.classes)           # List of class names
    print(data.class_to_idx)
    
    # create training and validation dataloaders to train the classifier
    if split_flag:
        val_split = 0.2 # validation split
        train_len = int((1.0 - val_split) * len(data))
        val_len = len(data) - train_len

        train_data, val_data = random_split(data, [train_len, val_len])

        print(len(train_data))
        print(len(val_data))
        
        train_data.dataset.transform = get_transforms_training()
        val_data.dataset.transform = get_transforms_validation()


        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader
    
    # create testing dataloader to evaluate the classfier
    else:
        data.transform = get_transforms_testing()
        train_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
        return train_loader






