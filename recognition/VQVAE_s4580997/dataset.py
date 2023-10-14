##################################
#
# Author: Joshua Wallace
# SID: 45809978
#
###################################

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import random
import torch
from utils import W, H

VQVAE_TRANSFORM = transforms.Compose([
    transforms.Resize((W, H)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])    

class Loader() :
    def __init__(self, batch_size = 32, path = './AD_NC/train', fraction=1.0, transform = VQVAE_TRANSFORM) :
        self.batch_size = batch_size
        self.path = path
        self.fraction = fraction
        self.dataset = None
        self.loader = None
        self.transform = transform
        self.var = None
    
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
        if self.var == None :
            self.var = self._get_variance()
        return self.var

    def _get_variance(self) -> float :
        # Initiating variables
        running_sum = 0.0
        running_sum_sq = 0.0
        total_pixels = 0
        
        for images, _ in self.loader:
            running_sum += torch.sum(images)
            running_sum_sq += torch.sum(images**2)
            
            total_pixels += torch.numel(images)
        
        mean = running_sum / total_pixels
        var = (running_sum_sq / total_pixels) - mean**2
        return var.item()
    
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


class ModelLoader() :
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    def __init__(self, model, batch_size = 32, path = './AD_NC/train', fraction=1.0, transform = VQVAE_TRANSFORM) :
        self.model = model
        self.batch_size = batch_size
        self.path = path
        self.fraction = fraction
        self.dataset = None
        self.loader = None
        self.transform = transform
        self.var = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
        if self.var == None :
            self.var = self._get_variance()
        return self.var

    def _get_variance(self) -> float :
        # Initiating variables
        running_sum = 0.0
        running_sum_sq = 0.0
        total_pixels = 0
        
        for images, _ in self.loader:
            running_sum += torch.sum(images)
            running_sum_sq += torch.sum(images**2)
            
            total_pixels += torch.numel(images)
        
        mean = running_sum / total_pixels
        var = (running_sum_sq / total_pixels) - mean**2
        return var.item()  # Returns the variance as a Python float
    
    def __len__(self):
        return len(self.loader)
    
    def __getitem__(self, index):
        img, _ = self.dataset[index]
        img = self.transform(img).unsqueeze(dim = 0)
        img = img.to(self.device)

        encoded = self.model.encoder(img)
        conv = self.model.conv_layer(encoded)
        _, _, _, encoding_indices = self.model.quantizer(conv)
        encoding_indices = encoding_indices.float().to(self.device)
        encoding_indices = encoding_indices.view(64, 64)
        encoding_indices = torch.stack([encoding_indices, encoding_indices, encoding_indices], dim = 0)
        return encoding_indices

    
class ModelDataset() :
    def __init__(self, model, batch_size = 32, root_dir = './AD_NC', fraction=1.0) :
        self.model = model
        self.train = ModelLoader(self.model, batch_size, f"{root_dir}/train", fraction)
        self.test = ModelLoader(self.model, batch_size, f"{root_dir}/test", fraction)
        
    
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
