"""
This file is used to load the dataset,
as well as calculate the mean and standard deviation of the dataset.

Author: Felix Hall 
Student number: 47028643
"""
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class ADNIDataset(Dataset):
    def __init__(self, root_dir, subset, transform=None):
        """
        Args:
 
            subset (string): 'train' or 'test' to specify which subset of the data to use.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image.

        """
        self.transform = transform
        self.root_dir = root_dir
        self.subset = subset
        self.transform = transform

        # build a list of all the file paths
        self.data_paths = []
        for label_class in ['AD', 'NC']:
            class_dir = os.path.join(root_dir, self.subset, label_class)
            for filename in os.listdir(class_dir):
                if os.path.isfile(os.path.join(class_dir, filename)):
                    self.data_paths.append((os.path.join(class_dir, filename), label_class))

       

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        img_path, label_class = self.data_paths[idx]
        image = Image.open(img_path).convert('L')  # Convert image to grayscale

        label = 1 if label_class == 'AD' else 0  # 1 for Alzheimer's (AC), 0 for normal control (NC)

        if self.transform:
            image = self.transform(image)

        return image, label



class DatasetStatistics:
    def __init__(self, dataset, batch_size=32):
        self.dataset = dataset
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        self.mean = 0.0
        self.std = 0.0

    def calculate_statistics(self):
        nb_samples = 0
        for data, _ in self.loader:
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            self.mean += data.mean(2).sum(0)
            nb_samples += batch_samples

        self.mean /= nb_samples

        for data, _ in self.loader:
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            self.std += ((data - self.mean.unsqueeze(1))**2).sum([0,2])
        
        self.std = torch.sqrt(self.std / (nb_samples * 256 * 256))  # Assuming image size is 256x256

        return self.mean, self.std

