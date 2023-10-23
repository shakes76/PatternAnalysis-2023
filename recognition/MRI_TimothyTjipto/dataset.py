'''Data loader for loading and preprocessing data'''

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,ConcatDataset,Dataset,TensorDataset, Subset
from PIL import Image
import random

# Load and return normalized data
def normalise_data(path, transform = None):

    # raw_dataset = datasets.ImageFolder(root=path)
    dataset = datasets.ImageFolder(root=path, transform=transform)
# Return the data set of Img, Label
    return dataset

def load_data(dataset,batch_size, num_worker = 0, shuffle=True):
    dataloader = DataLoader(dataset, batch_size=batch_size,num_workers=num_worker,shuffle=shuffle)
    return dataloader

# NOT in use
def filter_labels(dataset):
    images0,labels0 = [],[]
    images1,labels1 = [],[]
    for img,lbl in dataset:
        if lbl == 0:
            images0.append(img)
            labels0.append(lbl)
        else:
            images1.append(img)
            labels1.append(lbl)
    tensor_images0 = torch.stack(images0)
    tensor_labels0 = torch.tensor(labels0)

    tensor_images1 = torch.stack(images1)
    tensor_labels1 = torch.tensor(labels1)

    return TensorDataset(tensor_images0, tensor_labels0), TensorDataset(tensor_images1, tensor_labels1)

# # DO NOT SHUFFLE
# def make_pair(dataset1, dataset2):
    
#     # postive_pair1 = torch.cat((dataset1,dataset1),1)
#     # postive_pair2 = torch.cat((dataset2,dataset2),1)
#     # negative_pair1 = torch.cat((dataset1,dataset2),1)
#     # negative_pair2 = torch.cat((dataset2,dataset1),1)

#     postive_pair1 = ConcatDataset([dataset1,dataset2])

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


# Makes pairs of dataset
def make_pair(dataset1, dataset2):

    positive_pair1 = PairDataset(dataset1,dataset1,0)
    positive_pair2 = PairDataset(dataset2,dataset2,0)
    negative_pair1 = PairDataset(dataset1,dataset2,1)
    negative_pair2 = PairDataset(dataset2,dataset1,1)
    return positive_pair1,positive_pair2,negative_pair1,negative_pair2


def test_pair(test_dataset1, train_dataset1, test_dataset2, train_dataset2):
    test_positive_pair1 = PairDataset(test_dataset1,train_dataset1,0)
    test_positive_pair2 = PairDataset(test_dataset2,train_dataset2,0)
    test_negative_pair1 = PairDataset(test_dataset1,train_dataset2,1)
    test_negative_pair2 = PairDataset(test_dataset2,train_dataset1,1)
    return test_positive_pair1, test_positive_pair2, test_negative_pair1, test_negative_pair2


def shuffle(pos_pair1, pos_pair2, neg_pair1, neg_pair2):
    concatenated_dataset = ConcatDataset([pos_pair1, pos_pair2, neg_pair1, neg_pair2])
    return concatenated_dataset


# Gets dataset  with val size and train size Outputs the shuffle split train_set and val_set
def split_dataset(dataset, train_size, val_size):
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size,val_size])
    return  train_set,val_set 


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


def split_dataset_by_class(dataset):
    """
    Split a dataset into separate datasets for each class using PyTorch's Subset.

    Args:
    - dataset: The dataset to split.

    Returns:
    - dict of Subsets, where keys are class labels and values are the Subsets for that class.
    """

    class_datasets = {}

    # Get the class-to-index mapping from the dataset
    class_to_idx = dataset.class_to_idx

    for class_label, class_idx in class_to_idx.items():
        # Get the indices of samples for the current class
        indices = [i for i, (_, label) in enumerate(dataset) if label == class_idx]

        # Create a Subset for the current class
        class_subset = Subset(dataset, indices)
        class_datasets[class_label] = class_subset

    return class_datasets


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
