"""
Dataset loader
"""
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

class AlzheimerDataset(Dataset):
    """
    Dataset for loading Alzheimer's disease (AD) and Normal Control (NC) images.
    """
    def __init__(self, dataPath, transform=None):
        self.dataPath = dataPath
        self.AD, self.NC = None, None
        self.transform = transform
        self.loader()
    
    def loader(self):
        self.AD = self.loadPath("AD", label=1)
        self.NC = self.loadPath("NC", label=0)
        self.data = self.AD + self.NC

    def loadPath(self, type, label):
        arr = []
        for imageName in os.listdir(os.path.join(self.dataPath, type)):
            curImage = Image.open(os.path.join(self.dataPath, type, imageName)).convert("L")
            if self.transform:
                curImage = self.transform(curImage)
            arr.append((curImage, label))
        return arr

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def testLen(self):
        return (len(self.AD), len(self.NC))

if __name__ == "__main__":

    # EDA to see how many of each class exists in both train and test sets.
    test = AlzheimerDataset("AD_NC/test")
    train = AlzheimerDataset("AD_NC/train")
    print(train.testLen())
    print(test.testLen())