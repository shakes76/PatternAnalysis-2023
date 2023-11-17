"""
Imports Here
"""
"""numpy and torch"""
import numpy as np
import torch

"""PIL"""
from PIL import Image

"""torchvision and utils"""
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

"""os"""
import os

"""
Loading data from local file
"""
"""Assumes images have pixel values in range [0,255]"""
def getImages(trainDIRs, testDIRS):
    """Get image to tensor"""
    transform = transforms.Compose([
        transforms.PILToTensor()
    ])
    hflip = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.PILToTensor()
    ])
    vflip = transforms.Compose([
        transforms.RandomVerticalFlip(p=1.0),
        transforms.PILToTensor()
    ])
    dflip = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.PILToTensor()
    ])
    tlist = [transform, hflip, vflip, dflip]
    """Loading data into arrays"""
    xtrain, xtrain, xtest, ytest = [], [], [], []
    """training data"""
    size = [0, 0]
    for i, DIR in enumerate(trainDIRs):
        for t in tlist:
            px = []
            j = 0
            for filename in sorted(os.listdir(DIR)):
                f = os.path.join(DIR, filename)
                img = Image.open(f)
                tensor = t(img).float()
                tensor.require_grad = True
                px.append(tensor/255)
                j  = (j+1) % 20
                if j == 0:
                    xtrain.append(torch.stack(px))
                    px = []
                    size[i] += 1
    xtrain = torch.stack(xtrain)
    ytrain = torch.from_numpy(np.concatenate((np.ones(size[0]), np.zeros(size[1])), axis=0))

    """testing data"""
    size = [0, 0]
    for i, DIR in enumerate(testDIRs):
        for t in tlist:
            px = []
            j = 0
            for filename in sorted(os.listdir(DIR)):
                f = os.path.join(DIR, filename)
                img = Image.open(f)
                tensor = t(img).float()
                tensor.require_grad = True
                px.append(tensor/255)
                j = (j+1) % 20
                if j == 0:
                    xtest.append(torch.stack(px))
                    px = []
                    size[i] += 1
    xtest = torch.stack(xtest)
    idx = torch.randperm(xtest.size(0))
    xtest = xtest[idx, :]
    splitsize = int(xtest.shape[0]/2)
    xval, xtest = xtest.split(splitsize, dim=0)
    ytest = torch.from_numpy(np.concatenate((np.ones(size[0]), np.zeros(size[1])), axis=0))
    ytest = ytest[idx]
    yval, ytest = ytest.split(splitsize, dim=0)
    return xtrain, ytrain, xtest, ytest, xval, yval
"""
Dataloader
"""
class DatasetWrapper(Dataset):
    def __init__(self, X, y=None):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        else:
            return self.X[idx], self.y[idx]

trainDIRs = ['AD_NC/train/AD/', 'AD_NC/train/NC']
testDIRs = ['AD_NC/test/AD/', 'AD_NC/test/NC']
xtrain, ytrain, xtest, ytest, xval, yval = getImages(trainDIRs, testDIRs)
ytrain, ytest = ytrain.type(torch.LongTensor), ytest.type(torch.LongTensor)
xtrain = xtrain.permute(0, 2, 1, 3, 4)
xtest = xtest.permute(0, 2, 1, 3, 4)
xval = xval.permute(0, 2, 1, 3, 4)

def trainloader(batchsize=16):
    return DataLoader(DatasetWrapper(xtrain, ytrain), batch_size=batchsize, shuffle=True, pin_memory=True)

def valloader():
    return DataLoader(DatasetWrapper(xval, yval), batch_size=1, shuffle=True, pin_memory=True)

def testloader():
    return DataLoader(DatasetWrapper(xtest, ytest), batch_size=1, shuffle=True, pin_memory=True)

def trainshape():
    return xtrain.shape

def testshape():
    return xtest.shape