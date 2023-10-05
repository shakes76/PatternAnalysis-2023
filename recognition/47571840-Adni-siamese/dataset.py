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



ROOT_DIR_TRAIN = "/home/groups/comp3710/ADNI/AD_NC/train"


#-------- TRANSFORMERS FOR TRAINING AND TESTING -----------

def get_transforms_training():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Adjust these values if needed, for 1 channel
    ])


def get_transforms_testing():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Adjust these values if needed, for 1 channel
    ]) 

#--------- CREATE DATASET CLASS --------------------

class SiameseDataset(Dataset):
    def __init__(self, image_list, transforms=None):
        
        self.transforms = transforms
        self.image_list = image_list
        
        self.classes = list(set([label for _, label in image_list]))
        
        # Create pairs and labels
        self.pairs, self.labels = self.make_pairs()

    def make_pairs(self):
        # Referred and modified from: https://keras.io/examples/vision/siamese_contrastive/ 
        # Create a dictionary for each class and its corresponding indices
        class_indices = {cls: np.where(np.array([label for _, label in self.image_list]) == cls)[0] 
                 for cls in self.classes}
        
        pairs, labels = [], []
        seen_pairs = set()
        MAX_ATTEMPTS = 100
        
        for idx1, (img1, label1) in enumerate(self.image_list):   
            # Positive pair
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
            # Positive pair is labelled 0
            labels.append(0)

            # Negative pair
            label2 = random.choice([cls for cls in self.classes if cls != label1])
            idx2 = random.choice(class_indices[label2])
            pair = tuple(sorted([img1, self.image_list[idx2][0]]))
            attempts = 0
            while pair in seen_pairs and attempts < MAX_ATTEMPTS:
                idx2 = random.choice(class_indices[label2])
                pair = tuple(sorted([img1, self.image_list[idx2][0]]))
                attempts += 1
            if attempts == MAX_ATTEMPTS:
                continue
            
            seen_pairs.add(pair)
            pairs.append(list(pair))
             # Negative pair is labelled 1
            labels.append(1)
        
        return pairs, labels

    def __len__(self):
        return len(self.pairs)  # doubled because of pairs

    def __getitem__(self, idx):
        img1_path, img2_path = self.pairs[idx]
        
        # Open images and convert to RGB
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)

        # Apply transformations, if any
        if self.transforms:
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)

        return img1, img2, torch.tensor(self.labels[idx], dtype=torch.float32), img1_path, img2_path



#------------- FUNCTION TO BE CALLED TO RETURN DATALOADER -----------------


def create_siamese_dataloader(root_dir, batch_size=32, shuffle=True, transform=None, split_flag=True):
    data = datasets.ImageFolder(root=root_dir, transform=None)
    print("Total Number of images:", len(data))

    if split_flag:

        val_split=0.2
        train_len = int((1.0 - val_split) * len(data))
        val_len = len(data) - train_len

        train_data, val_data = random_split(data, [train_len, val_len])

        train_image_list = [(data.imgs[i][0], data.targets[i]) for i in train_data.indices]
        val_image_list = [(data.imgs[i][0], data.targets[i]) for i in val_data.indices]

        train_siamese_dataset = SiameseDataset(train_image_list, transforms=transform)
        val_siamese_dataset = SiameseDataset(val_image_list, transforms=transform)

        print("Training Pairs Length:", len(train_siamese_dataset))
        print("Validation Pairs Length:", len(val_siamese_dataset))

        train_loader = DataLoader(train_siamese_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_siamese_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader
    else:
        siamese_dataset = SiameseDataset(data.imgs, transforms=transform)
        data_loader = DataLoader(siamese_dataset, batch_size=batch_size, shuffle=shuffle)
        return data_loader






############ CHECKING IF DATALOADER WORKS ######################

ROOT_DIR_TRAIN = "/home/groups/comp3710/ADNI/AD_NC/train"
train_loader, val_loader = create_siamese_dataloader(ROOT_DIR_TRAIN, batch_size=32, transform=get_transforms_training(), split_flag=True)


# Get the first batch from the train_loader
# first_batch = next(iter(train_loader))

# img1_batch, img2_batch, labels_batch, img1_path_batch, img2_path_batch = first_batch

# # Display tensors, labels, and paths of the first two images
# for i in range(4):
#     print(f"Image {i + 1} Tensor:\n", img1_batch[i], "\n")
#     print(f"Image {i + 1} Path:", img1_path_batch[i], "\n")
#     print(f"Image {i + 1} Pair Tensor:\n", img2_batch[i], "\n")
#     print(f"Image {i + 1} Pair Path:", img2_path_batch[i], "\n")
#     print(f"Label {i + 1}:", labels_batch[i].item(), "\n")
#     print("-----------------------------\n")

######### CHECK FOR LEAKAGE ########################
# def check_for_leakage_from_loaders(train_loader, val_loader):
#     # 1. Extract image paths from both loaders
#     train_image_paths = []
#     for (img1_batch, img2_batch, _, img1_path_batch, img2_path_batch) in train_loader:
#         train_image_paths.extend(img1_path_batch)
#         train_image_paths.extend(img2_path_batch)
#     train_image_paths = set(train_image_paths)

#     val_image_paths = []
#     for (img1_batch, img2_batch, _, img1_path_batch, img2_path_batch) in val_loader:
#         val_image_paths.extend(img1_path_batch)
#         val_image_paths.extend(img2_path_batch)
#     val_image_paths = set(val_image_paths)


#     # 2. Check for image overlaps
#     common_images = train_image_paths.intersection(val_image_paths)
#     if common_images:
#         print(f"Found {len(common_images)} overlapping images between train and val sets!")
#     else:
#         print("No overlapping images between train and val sets.")
    
#     # 3. Check for pair overlaps in Siamese dataset
#     train_pairs = set()
#     for (img1_batch, img2_batch, _, img1_path_batch, img2_path_batch) in train_loader:
#         for img1_path, img2_path in zip(img1_path_batch, img2_path_batch):
#             train_pairs.add(tuple(sorted([img1_path, img2_path])))

#     val_pairs = set()
#     for (img1_batch, img2_batch, _, img1_path_batch, img2_path_batch) in val_loader:
#         for img1_path, img2_path in zip(img1_path_batch, img2_path_batch):
#             val_pairs.add(tuple(sorted([img1_path, img2_path])))

#     common_pairs = train_pairs.intersection(val_pairs)
#     if common_pairs:
#         print(f"Found {len(common_pairs)} overlapping pairs between train and val sets!")
#     else:
#         print("No overlapping pairs between train and val sets.")

# # After creating your dataloaders, call the function:
# check_for_leakage_from_loaders(train_loader, val_loader)





