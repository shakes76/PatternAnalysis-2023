'''Data loader for loading and preprocessing data'''

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,ConcatDataset,Dataset,TensorDataset, Subset
from PIL import Image
import random

def get_transform():
    """
    Returns a composed transformation for datasets.
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale with one channel
        transforms.Resize((120,128)),  # Resize to (120,128)
        transforms.ToTensor(),
        # You can add more transformations if needed
    ])
    return transform

def get_dataloader(dataset, batch_size = 16, shuffle=True):
    v_dataloader = DataLoader(dataset,
                            shuffle=shuffle,
                            num_workers=1,
                            batch_size=batch_size)
    return v_dataloader

# Pairs up dataset
class PairDataset(Dataset):
    def __init__(self,dataset1,dataset2, label):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.label = label

    def __len__(self):
        return min(len(self.dataset1), len(self.dataset2))
    
    def __getitem__(self, index):
        img1,_ = self.dataset1[index]
        img2,_ = self.dataset2[index]

        return img1, img2, self.label


def visualise_1(dataset):
    img,lab = random.choice(dataset)
    plt.title(lab)
    plt.imshow(img)
    plt.axis('off')
    plt.savefig("visualise_1")
    plt.show()

    
def visualise_batch(dataloader):
    LABELS = ['POS','NEG']

    example_batch = iter(dataloader)
    images1,images2,labels = next(example_batch)

    plt.figure(figsize=(16,4)) # width x height
    batch_size = len(images1)
    for idx in range(batch_size):

        image1 = transforms.ToPILImage()(images1[idx])
        image2 = transforms.ToPILImage()(images2[idx])
        label = LABELS[int(labels[idx].item())]

        plt.subplot(2,batch_size,idx+1)
        
        plt.imshow(image1,cmap='gray')
        plt.axis('off')

        plt.subplot(2,batch_size,idx+1+batch_size)
        plt.imshow(image2,cmap='gray')
        plt.title(label)
        plt.axis('off')

    plt.savefig("visualise_batch")
    plt.show()



class SiameseNetworkDataset1(Dataset):
    def __init__(self,imageFolderDataset,transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform

    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)

        #We need to approximately 50% of images to be in the same class
        should_get_same_class = random.randint(0,1)
        if should_get_same_class:
            while True:
                #Look untill the same class image is found
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


class SiameseNetworkDataset_test(Dataset):
    def __init__(self,imageFolderDataset,transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform

    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)

        #We need to approximately 50% of images to be in the same class
        should_get_same_class = random.randint(0,1)
        if should_get_same_class:
            while True:
                #Look untill the same class image is found
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

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32)), img0_tuple[1],img1_tuple[1]

    def __len__(self):
        return len(self.imageFolderDataset.imgs)
