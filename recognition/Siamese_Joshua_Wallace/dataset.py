import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import random
import torch

"""
"""
class Dataset():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    def __init__(self, batch_size = 32, root_dir = './AD_NC') :
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.train_loader = None
        self.test_loader = None
    
    def load_train(self) -> None:
        train_dataset = ImageFolder(root=f"{self.root_dir}/train")
        siamese_train_dataset = ADNIDataset(train_dataset, transform=self.transform)
        self.train_loader = DataLoader(siamese_train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def get_train(self) -> DataLoader :
        if self.train_unloaded() :
            print('Retrieving trainset.')
            self.load_train()
        return self.train_loader

    def load_test(self) -> None :
        test_dataset = ImageFolder(root=f"{self.root_dir}/test")
        siamese_test_dataset = ADNIDataset(test_dataset, transform=self.transform)
        self.test_loader = DataLoader(siamese_test_dataset, batch_size=self.batch_size, shuffle=True)
    
    def get_test(self) -> DataLoader :
        if self.test_unloaded() :
            print('Retrieving testset.')
            self.load_test()
        return self.test_loader
    
    def train_unloaded(self) -> bool :
        return not self.train_loader
    
    def test_unloaded(self) -> bool :
        return not self.test_loader

class ADNIDataset(Dataset):
    def __init__(self, imagefolder_dataset, transform):
        self.imagefolder_dataset = imagefolder_dataset
        self.transform = transform

    def __len__(self):
        return len(self.imagefolder_dataset)

    def __getitem__(self, index):
        img1, label1 = self.imagefolder_dataset[index]
        # Get another image from the same class
        while True:
            index2 = random.choice(range(len(self.imagefolder_dataset)))
            img2, label2 = self.imagefolder_dataset[index2]
            if label1 == label2:
                break

        img1 = self.transform(img1)
        img2 = self.transform(img2)
        return (img1, img2), torch.tensor(int(label1 == label2), dtype=torch.float32)
