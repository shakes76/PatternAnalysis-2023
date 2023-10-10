import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
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

    def __init__(self, batch_size = 32, root_dir = './AD_NC', fraction=1.0) :
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.fraction = fraction
        self.train_loader = None
        self.test_loader = None
    
    def load_train(self) -> None:
        train_dataset = ImageFolder(root=f"{self.root_dir}/train")
        train_dataset = self._get_subset(train_dataset)
        siamese_train_dataset = ADNIDataset(train_dataset, transform=self.transform)
        self.train_loader = DataLoader(siamese_train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def _get_subset(self, dataset):
        num_samples = int(len(dataset) * self.fraction)
        indices = random.sample(range(len(dataset)), num_samples)
        return Subset(dataset, indices)
    
    def get_train(self) -> DataLoader :
        if self.train_unloaded() :
            print('Retrieving trainset.')
            self.load_train()
        return self.train_loader

    def load_test(self) -> None :
        test_dataset = ImageFolder(root=f"{self.root_dir}/test")
        test_dataset = self._get_subset(test_dataset)
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
    def __init__(self, batch_size, root_dir, fraction):
        super().__init__(batch_size, root_dir, fraction)

    def __len__(self):
        return len(super.root_dir)

    def __getitem__(self, index):
        img, label = super.root_d[index]

        img = self.transform(img)
        return img