import torch

import torchvision.transforms as transforms

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_paths[idx])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image

class TripletDataset(Dataset):
    def __init__(self, AD, NC, transform=None):
        self.X = AD + NC
        self.AD = AD
        self.NC = NC
        self.Y = torch.cat((torch.ones(len(AD)), torch.zeros(len(NC))), dim=0)
        self.anc_indices = torch.randperm(len(self.X))
        self.pos_indices = torch.randperm(len(self.X)) % len(AD)
        self.neg_indices = torch.randperm(len(self.X)) % len(NC)
        self.transform = transform

    def __len__(self):
        return len(self.anc_indices)

    def __getitem__(self, idx):
        anc = self.anc_indices[idx]
        pos = self.pos_indices[idx]
        neg = self.neg_indices[idx]
        img1 = self.X[anc]
        img2 = self.AD[pos]
        img3 = self.NC[neg]
        label = self.Y[anc]

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        return img1, img2, img3, torch.tensor([1 - label, label])




def intensity_normalization(img, mean = None, std = None):
    mean = torch.mean(img)
    std = torch.std(img)
    return (img - mean) / std

class CustomNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        return (img - self.mean) / self.std

transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),  # Random affine transformations with smaller parameters
    transforms.Lambda(intensity_normalization)
])

transform_test = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Lambda(intensity_normalization)
])

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        return (img - self.mean) / self.std



def gen_loaders(root_dir = '', batch_size = 96):
    loaders = {}
    AD_train = CustomDataset(root_dir=root_dir + os.path.join('train', 'AD'), transform=transform_train)
    NC_train = CustomDataset(root_dir=root_dir + os.path.join('train', 'NC'), transform=transform_train)

    X = torch.stack([img for img in AD_train + NC_train])
    mean = X.mean()
    std = X.std()
    normalize = transforms.Compose([
            Normalize(mean, std)
        ])
    train_dataset = TripletDataset(AD_train, NC_train, normalize)


    # Create DataLoaders for the two parts
    loaders['train'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    AD_test = CustomDataset(root_dir=os.path.join('test', 'AD'), transform=transform_test)
    NC_test = CustomDataset(root_dir=os.path.join('test', 'NC'), transform=transform_test)

    test_dataset = TripletDataset(AD_test, NC_test, normalize)

    loaders['test'] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return loaders