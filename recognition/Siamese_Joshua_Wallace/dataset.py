import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import random
import torch
from abc import ABC, abstractmethod

VQVAE_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class Loader() :
    def __init__(self, batch_size = 32, path = './AD_NC/train', fraction=1.0, transform = VQVAE_TRANSFORM) :
        self.batch_size = batch_size
        self.path = path
        self.fraction = fraction
        self.dataset = None
        self.loader = None
        self.transform = transform
    
    def load(self) -> None:
        dataset = ImageFolder(root=f"{self.path}", transform=self.transform)
        self.dataset = self._get_subset(dataset)
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
    
    def _get_subset(self, dataset):
        num_samples = int(len(dataset) * self.fraction)
        indices = random.sample(range(len(dataset)), num_samples)
        return Subset(dataset, indices)
    
    def get(self) -> DataLoader :
        if self.unloaded() :
            print('Retrieving dataset.')
            self.load()
        return self.loader

    def unloaded(self) -> bool :
        return not self.loader
    
    def variance(self) -> float :
        if self.unloaded() :
            print('Retrieving dataset.')
            self.load()
        return torch.var(self.loader.dataset.tensors[0])
    
    def __len__(self):
        return len(self.loader)
    
    def __getitem__(self, index):
        img, _ = self.dataset[index]
        img = self.transform(img)
        return img
    
class Dataset():
    def __init__(self, batch_size = 32, root_dir = './AD_NC', fraction=1.0) :
        self.train = Loader(batch_size, f"{root_dir}/train", fraction)
        self.test = Loader(batch_size, f"{root_dir}/test", fraction)
    
    def load_train(self) -> None:
        self.train.load()
    
    def get_train(self) -> DataLoader :
        return self.train.get()

    def load_test(self) -> None :
        self.test.load()
    
    def get_test(self) -> DataLoader :
        return self.test.get()

    def train_unloaded(self) -> bool :
        return self.train.unloaded()
    
    def test_unloaded(self) -> bool :
        return self.test.unloaded()
    
    def train_var(self) -> float :
        return self.train.variance()
    
    def test_var(self) -> float :
        return self.test.variance()
