"""
Name: dataset.py
Student: Ethan Pinto (s4642286)
Description: Containing the data loader for loading and preprocessing your data.
"""

import torch
import random
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader

dataroot = "/home/groups/comp3710"

data_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

ADNI_brain_data = datasets.ImageFolder(root=dataroot,
                                           transform=data_transform)

dataset_loader = DataLoader(ADNI_brain_data, batch_size=4,
                            shuffle=True, num_workers=4)

class ADNI_Dataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
    
    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)

    #We need to approximately 50% of images to be in the same class
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                #Look until the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:

            while True:
                #Look untill a different class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)