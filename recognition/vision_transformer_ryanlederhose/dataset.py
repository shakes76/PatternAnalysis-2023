from torch.utils.data import DataLoader as TorchDataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import random_split

class DataLoader(object):
    def __init__(self, batch_size=64) -> None:
        self.trainingFile = "AD_NC/train/"
        self.testFile = "AD_NC/test/"
        self.imageSize = 192
        self.batchSize = batch_size
        self.normalisation = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        self.trainloader = None
        self.testloader = None
        self.validloader = None
        
        self.load_training_data()
        self.load_test_data()

    def load_training_data(self):
        # Get training dataset from image folder
        train_images = ImageFolder(root=self.trainingFile, transform=T.Compose([T.Resize(self.imageSize), 
                                    T.CenterCrop(self.imageSize), T.ToTensor(), T.Normalize(*self.normalisation)])
                                    )
        
        # Get data loader of training folder
        original_loader = TorchDataLoader(train_images, batch_size=self.batchSize, 
                                 shuffle=True, num_workers=3, pin_memory=True)

        # Define the split ratio or a fixed number of validation samples
        split_ratio = 0.8

        # Calculate the sizes for training and validation sets
        total_samples = len(original_loader.dataset)
        train_size = int(split_ratio * total_samples)
        valid_size = total_samples - train_size

        # Use the original DataLoader's dataset for splitting
        train_dataset, valid_dataset = random_split(
            original_loader.dataset, [train_size, valid_size]
        )

        # Create DataLoader for training dataset
        self.trainloader = TorchDataLoader(
            train_dataset,
            batch_size=self.batchSize,
            shuffle=True,  # You can shuffle the training data
            num_workers=3  # Number of worker processes for loading data
        )

        # Create DataLoader for the validation dataset
        self.validloader = TorchDataLoader(
            valid_dataset,
            batch_size=self.batchSize,
            shuffle=False,  # Validation data shouldn't be shuffled
            num_workers=3  # Number of worker processes for loading data
        )

        
    def load_test_data(self):
        # Get training dataset from image folder
        test_images = ImageFolder(root=self.testFile, transform=T.Compose([T.Resize(self.imageSize), 
                                    T.CenterCrop(self.imageSize), T.ToTensor(), T.Normalize(*self.normalisation)]),
                                    )

        # Get training loader
        self.testloader = TorchDataLoader(test_images, batch_size=self.batchSize, 
                                 shuffle=True, num_workers=3, pin_memory=True)
        
    def get_training_loader(self):
        return self.trainloader
    
    def get_test_loader(self):
        return self.testloader
    
    def get_valid_loader(self):
        return self.validloader

