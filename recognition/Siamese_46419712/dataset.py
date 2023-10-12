from tabnanny import check
import torchvision
import torch
import torchvision.transforms as transforms
import random
import numpy as np
import matplotlib.pyplot as plt

TRAIN_PATH = "/home/groups/comp3710/ADNI/AD_NC/train"
TRAIN_PATH = "./AD_NC/train"

TEST_PATH = "/home/groups/comp3710/ADNI/AD_NC/train"
TEST_PATH = "./AD_NC/train"

class PairedDataset(torch.utils.data.Dataset):
    
    def __init__(self, trainset):
        # follow this source: https://datahacker.rs/019-siamese-network-in-pytorch-with-application-to-face-similarity/
        self.trainset = trainset

    def __getitem__(self, index):
        img0, label0 = self.trainset[index]

        check_same_class = random.randint(0,1) 
        
        while True:
            img1, label1 = random.choice(self.trainset)

            if not torch.equal(img0, img1):

                if label1 == label0 and check_same_class:
                    break
                elif label1 != label0 and not check_same_class:
                    break
    
    def __len__(self):
        return len(self.trainset)

def split_dataset(dataset, seed=False):

    if seed:
        generator = torch.Generator().manual_seed(35)
    else:
        generator = torch.Generator()

    # follow the principle -> 90% train 10% val
    train_set, val_set = torch.utils.data.random_split(dataset, [0.9, 0.1], generator=generator)

    return train_set, val_set

def load_train_data():
    path = TRAIN_PATH

    # torch.manual_seed(33) # for reproduce in the future
    # Data transformation
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    load_train_image = torchvision.datasets.ImageFolder(root=path, transform=transform_train)

    train_image, val_image = split_dataset(load_train_image)

    trainset = PairedDataset(train_image)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

    valset = PairedDataset(val_image)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=True, num_workers=0)

    return train_loader, val_loader

def load_train_data_classifier():
    path = TRAIN_PATH

    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.ImageFolder(root=path, transform=transform_train)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

    return train_loader

def load_test_data():
    path = TEST_PATH

    # Data transformation
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # standard scaling for normalize, doesn't know much on the status of the entire dataset
    ])

    testset = torchvision.datasets.ImageFolder(root=path, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=0)
    return test_loader

if __name__ == '__main__':
    # test dataset
    testloader = load_train_data()
