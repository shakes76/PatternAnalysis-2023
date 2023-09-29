import os
import numpy as np
import random
from torchvision import datasets, transforms
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from PIL import Image



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
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.data = datasets.ImageFolder(root=self.root_dir, transform=None)  # Note: Don't apply transforms here
        
        # print(self.data.classes)  # ['AD', 'NC'] or ['NC', 'AD'] depending on folder ordering
        print(self.data.class_to_idx)
        print("Total Number of images:",  len(self.data))


        self.classes = self.data.classes
        
        # Create pairs and labels
        self.pairs, self.labels = self.make_pairs()

    def make_pairs(self):
        # Referenced from: https://keras.io/examples/vision/siamese_contrastive/
        num_classes = len(self.classes)
        
        # Create a dictionary for each class and its corresponding indices
        class_indices = {idx: np.where(np.array(self.data.targets) == idx)[0] 
                 for idx in self.data.class_to_idx.values()}

        # print(class_indices)
        
        pairs, labels = [], []
        
        for idx1, data1 in enumerate(self.data.imgs):
            img1, label1 = data1
            
            # Positive pair
            idx2 = random.choice(class_indices[label1])
            img2 = self.data.imgs[idx2][0]
            pairs.append([img1, img2])
            # The label 0 denotes that the two images in the pair are from the same class.
            labels.append(0)
            
            # Negative pair
            label2 = random.choice([cls for cls in range(num_classes) if cls != label1])
            idx2 = random.choice(class_indices[label2])
            img2 = self.data.imgs[idx2][0]
            pairs.append([img1, img2])
            # The label 1 denotes that the two images in the pair are from the different class.
            labels.append(1)
        
        return pairs, labels

    def __len__(self):
        return len(self.data.imgs) * 2  # doubled because of pairs

    def __getitem__(self, idx):
        img1_path, img2_path = self.pairs[idx]
        
        # Open images and convert to RGB
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)

        # Apply transformations, if any
        if self.transforms:
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)

        return img1, img2, torch.tensor(self.labels[idx], dtype=torch.float32)



#------------- FUNCTION TO BE CALLED TO RETURN DATALOADER -----------------

def create_siamese_dataloader(root_dir, batch_size=32, shuffle=True,transform=None):
    siamese_dataset = SiameseDataset(root_dir, transforms=transform)
    dataloader = DataLoader(siamese_dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader






############ CHECKING IF DATALOADER WORKS ######################
# dataloader = create_siamese_dataloader(ROOT_DIR_TRAIN, batch_size=4, transform=get_transforms_training())
# num_batches_to_view = 2

# for i, (img1_batch, img2_batch, labels_batch) in enumerate(dataloader):
#     if i >= num_batches_to_view:
#         break
#     print(f"Batch {i+1}")
   
#     print("Labels:",labels_batch )
#     print(img1_batch.size(), img2_batch.size(), labels_batch.size())
#     print("------")




