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

# Define the transform globally
VQVAE_TRANSFORM = transforms.Compose([
    transforms.Resize((W, H)),
    transforms.ToTensor(),
])    

class Loader() :
    """
    Dataset loader which allows for the splitting of the dataset into a subset.
    """
    def __init__(self, batch_size = 32, path = './AD_NC/train', fraction=1.0, transform = VQVAE_TRANSFORM) :
        """
        Initialise the loader with the given parameters.

        Parameters
        ----------
        param1 : batch_size
            Batch size for the loader.
        param2 : path
            Path to the dataset.
        param3 : fraction
            Fraction of the dataset to use.
        param4 : transform
            Transform to apply to the dataset.
        """
        self.batch_size = batch_size
        self.path = path
        self.fraction = fraction
        self.dataset = None
        self.loader = None
        self.transform = transform
        self.var = None
    
    def load(self) -> None:
        """
        Load the dataset into the loader.
        """
        dataset = ImageFolder(root=f"{self.path}", transform=self.transform)
        self.dataset = self._get_subset(dataset)
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
    
    def _get_subset(self, dataset):
        """
        Get a subset of the dataset based on the fraction parameter.
        """
        num_samples = int(len(dataset) * self.fraction)
        indices = random.sample(range(len(dataset)), num_samples)
        return Subset(dataset, indices)
    
    def get(self) -> DataLoader :
        """
        Retrieve the loader, if unloaded, then load it.
        """
        if self.unloaded() :
            print('Retrieving dataset.')
            self.load()
        return self.loader

    def unloaded(self) -> bool :
        """
        Check if the loader is unloaded.
        """
        return not self.loader
    
    def variance(self) -> float :
        """
        Get the variance of the dataset.
        """

        if self.unloaded() :
            print('Retrieving dataset.')
            self.load()
        if self.var == None :
            self.var = self._get_variance()
        return self.var

    def _get_variance(self) -> float :
        """
        Calculate the variance of the dataset.
        """
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
        """
        Get the length of the loader.
        """
        return len(self.loader)
    
    def __getitem__(self, index):
        """
        Get an item from the loader.
        """
        img, _ = self.dataset[index]
        img = self.transform(img)
        return img
    
class Dataset():
    """
    The dataset class which contains train and test loaders.
    """
    def __init__(self, batch_size = 32, root_dir = './AD_NC', fraction=1.0) :

        self.train = Loader(batch_size, f"{root_dir}/train", fraction)
        self.test = Loader(batch_size, f"{root_dir}/test", fraction)
    
    def load_train(self) -> None:
        """
        Load the training dataset.
        """
        self.train.load()
    
    def get_train(self) -> DataLoader :
        """
        Get the training loader.
        """
        return self.train.get()

    def load_test(self) -> None :
        """
        Load the test dataset.
        """
        self.test.load()
    
    def get_test(self) -> DataLoader :
        """
        Get the test loader.
        """
        return self.test.get()

    def train_unloaded(self) -> bool :
        """
        Check if the training loader is unloaded.
        """
        return self.train.unloaded()
    
    def test_unloaded(self) -> bool :
        """
        Check if the test loader is unloaded.
        """
        return self.test.unloaded()
    
    def train_var(self) -> float :
        """
        Get the variance of the training dataset.
        """
        return self.train.variance()
    
    def test_var(self) -> float :
        """
        Get the variance of the test dataset.
        """
        return self.test.variance()


class ModelLoader() :
    """
    ModelLoader class is used to generate the loader from a given model. For this application, it will generate
    the dataset from the VQVAE model.
    """
    transform = VQVAE_TRANSFORM

    def __init__(self, model, batch_size = 32, path = './AD_NC/train', fraction=1.0, transform = VQVAE_TRANSFORM) :
        """
        Initialise the loader for a model.

        Parameters
        ----------
        param1 : model
            Model to generate the loader from. Must be of class VQVAE.
        param2 : batch_size
            Batch size for the loader.
        param3 : path
            Path to the dataset.
        param4 : fraction
            Fraction of the dataset to use.
        param5 : transform
            Transform to apply to the dataset.
        """
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
        """
        Load the dataset into the loader from the path.
        """
        dataset = ImageFolder(root=f"{self.path}", transform=self.transform)
        self.dataset = self._get_subset(dataset)
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
    
    def _get_subset(self, dataset):
        """
        Get a subset of the dataset based on the fraction parameter.
        """
        num_samples = int(len(dataset) * self.fraction)
        indices = random.sample(range(len(dataset)), num_samples)
        return Subset(dataset, indices)
    
    def get(self) -> DataLoader :
        """
        Retrieve the loader, if unloaded, then load it.
        """
        if self.unloaded() :
            self.load()
        return self.loader

    def unloaded(self) -> bool :
        """
        Check if the loader is unloaded.
        """
        return not self.loader
    
    def variance(self) -> float :
        """
        Get the variance of the dataset.
        """
        if self.unloaded() :
            self.load()
        if self.var == None :
            self.var = self._get_variance()
        return self.var

    def _get_variance(self) -> float :
        """
        Calculate the variance of the dataset.
        """
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
        """
        Get the length of the loader.
        """
        return len(self.loader)
    
    def __getitem__(self, index):
        """
        Get an item from the loader. The values are encoded and then quantized according to the provided model.
        """
        img, _ = self.dataset[index]
        img = self.transform(img)
        img = img.unsqueeze(dim = 0)
        img = img.to(self.device)
        encoded = self.model.encoder(img)
        conv = self.model.conv(encoded)
        _, _, _, encoding, encoding_indices = self.model.quantizer(conv)
        encoding_indices = encoding_indices.float().to(self.device)
        encoding_indices = encoding_indices.view(64, 64)
        encoding_indices = torch.stack([encoding_indices, encoding_indices, encoding_indices], dim = 0)
        return encoding_indices

class ModelDataset() :
    """
    ModelDataset class is used to generate the dataset from a given model for train and test loaders
    """
    def __init__(self, model, batch_size = 32, root_dir = './AD_NC', fraction=1.0) :
        """
        Initialise the dataset for a model.

        Parameters
        ----------
        param1 : model
            Model to generate the loader from. Must be of class VQVAE.
        param2 : batch_size
            Batch size for the loader.
        param3 : path
            Path to the dataset, assuming folders named train and test.
        param4 : fraction
            Fraction of the dataset to use.
        """
        self.model = model
        self.train = ModelLoader(self.model, batch_size, f"{root_dir}/train", fraction)
        self.test = ModelLoader(self.model, batch_size, f"{root_dir}/test", fraction)
        
    def load_train(self) -> None:
        """
        Load the training dataset.
        """
        self.train.load()
    
    def get_train(self) -> DataLoader :
        """
        Get the training loader.
        """
        return self.train.get()

    def load_test(self) -> None :
        """
        Load the test dataset.
        """
        self.test.load()
    
    def get_test(self) -> DataLoader :
        """
        Get the test loader.
        """
        return self.test.get()

    def train_unloaded(self) -> bool :
        """
        Check if the training loader is unloaded.
        """
        return self.train.unloaded()
    
    def test_unloaded(self) -> bool :
        """
        Check if the test loader is unloaded.
        """
        return self.test.unloaded()
    
    def train_var(self) -> float :
        """
        Get the variance of the training dataset.
        """
        return self.train.variance()
    
    def test_var(self) -> float :
        """
        Get the variance of the test dataset.
        """
        return self.test.variance()    
