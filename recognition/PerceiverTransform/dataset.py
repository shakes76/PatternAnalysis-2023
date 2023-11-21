import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import random_split
from PIL import Image

class AlzheimerDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
    # Initialzing the dataset with parameters (root directory with entire dataset present)
    
        self.root_dir = os.path.join(root_dir, mode)  # This appends 'train' or 'test' based on mode, effectively separating the datasets.
        self.transform = transform
        # Extracts the filepaths for each dataset type. 
        self.nc_images = [os.path.join(self.root_dir, 'NC', img) for img in os.listdir(os.path.join(self.root_dir, 'NC')) if img.endswith('.jpeg')]
        self.ad_images = [os.path.join(self.root_dir, 'AD', img) for img in os.listdir(os.path.join(self.root_dir, 'AD')) if img.endswith('.jpeg')]
        self.total_images = self.nc_images + self.ad_images
        

    def __len__(self):
        return len(self.total_images)
    # Returns total number of images

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
    # Returns the images index 
        img_name = self.total_images[idx]
        image = Image.open(img_name)
        
        # Assigning a label based on the image type.
        label = 0 if img_name in self.nc_images else 1  #0 for NC and 1 for AD
        
        # Applying tranformation if needed.
        if self.transform:
            image = self.transform(image)

        return image, label

def get_dataloaders(root_dir, batch_size):

    # Defining image transformations for preprocessing purpose i.e resizing and converting to tensor.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resizing to fit typical CNN input sizes. 
        transforms.ToTensor(),
    ])

    # The AlzheimerDataset class is instantiated twice, dataset for training and testing purposes. 
    train_dataset = AlzheimerDataset(root_dir, mode='train', transform=transform)
    test_dataset = AlzheimerDataset(root_dir, mode='test', transform=transform)

    # Splitting the training dataset into train and validation subsets using a 65-35 split.
    train_length = int(0.65 * len(train_dataset))
    valid_length = len(train_dataset) - train_length
    train_subset, valid_subset = random_split(train_dataset, [train_length, valid_length])

    # Dataloaders created for the train, validation, and test datasets with specific batch sizes.
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader