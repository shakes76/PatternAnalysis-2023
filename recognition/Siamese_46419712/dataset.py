import torchvision
import torch
import torchvision.transforms as transforms
import random
import numpy as np
import matplotlib.pyplot as plt


class PairedDataset(torch.utils.data.Dataset):
    
    def __init__(self, trainset):
        # follow this source: https://datahacker.rs/019-siamese-network-in-pytorch-with-application-to-face-similarity/
        self.trainset = trainset

    def __getitem__(self, index):
        img0, label0 = self.trainset[index]

        check_same_class = random.randint(0,1) 
        if check_same_class:
            while True:
                #Look untill the same class image is found
                img1, label1 = random.choice(self.trainset)
                if torch.equal(img0, img1):
                    continue

                if label1 == label0:
                    break
        else:
            while True:
                #Look untill a different class image is found
                img1, label1 = random.choice(self.trainset)
                
                if torch.equal(img0, img1):
                    continue
                
                if label1 != label0:
                    break
        
        return img0, img1, torch.from_numpy(np.array([int(label0 != label1)], dtype=np.float32))
    
    def __len__(self):
        return len(self.trainset)

def split_dataset(dataset, seed=True):

    if seed:
        generator = torch.Generator().manual_seed(35)
    else:
        generator = torch.Generator()

    # follow the principle -> 80% train 20% val
    train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=generator)

    return train_set, val_set

def load_train_data():
    path="/home/groups/comp3710/ADNI/AD_NC/train"
    # this for testing
    path="./AD_NC/train"

    torch.manual_seed(33) # for reproduce in the future
    # Data transformation
    transform_train = transforms.Compose([
        transforms.Resize(105),
        transforms.CenterCrop(105),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # transform_train = transforms.Compose([
    #     transforms.Resize(105),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.CenterCrop(105),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    load_train_image = torchvision.datasets.ImageFolder(root=path, transform=transform_train)

    train_image, val_image = split_dataset(load_train_image)

    trainset = PairedDataset(train_image)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

    valset = PairedDataset(val_image)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=True, num_workers=0)

    return train_loader, val_loader

def load_train_data_classifier():
    path="/home/groups/comp3710/ADNI/AD_NC/train"
    path="./AD_NC/train"

    transform_train = transforms.Compose([
        transforms.Resize(105),
        transforms.CenterCrop(105),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.ImageFolder(root=path, transform=transform_train)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

    return train_loader

def load_test_data():
    path="/home/groups/comp3710/ADNI/AD_NC/test"
    # this for testing
    path="./AD_NC/test"
    # dataset = torchvision.datasets.ImageFolder(root=path, transform=transforms.ToTensor())

    # # Calculate the mean and standard deviation
    # mean = torch.mean(dataset.data.float() / 255.0)
    # std = torch.std(dataset.data.float() / 255.0)

    # Data transformation
    transform_test = transforms.Compose([
        transforms.Resize(105),
        transforms.CenterCrop(105),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # standard scaling for normalize, doesn't know much on the status of the entire dataset
    ])

    testset = torchvision.datasets.ImageFolder(root=path, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=0)
    return test_loader

if __name__ == '__main__':
    # test dataset
    testloader = load_train_data()
