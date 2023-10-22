import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import random_split
from PIL import Image

class AlzheimerDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):

        self.root_dir = os.path.join(root_dir, mode)  # This appends 'train' or 'test' based on mode, effectively separating the datasets.
        self.transform = transform
        self.nc_images = [os.path.join(self.root_dir, 'NC', img) for img in os.listdir(os.path.join(self.root_dir, 'NC')) if img.endswith('.jpeg')]
        self.ad_images = [os.path.join(self.root_dir, 'AD', img) for img in os.listdir(os.path.join(self.root_dir, 'AD')) if img.endswith('.jpeg')]
        self.total_images = self.nc_images + self.ad_images

    def __len__(self):
        return len(self.total_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.total_images[idx]
        image = Image.open(img_name)
        
        label = 0 if img_name in self.nc_images else 1  # 0 for NC and 1 for AD
        
        if self.transform:
            image = self.transform(image)

        return image, label

def get_dataloaders(root_dir, batch_size):

    # Image preprocessing steps: resizing to a common size, converting to tensor, and normalizing.
    # Adjust these transformations based on your specific needs.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resizing to fit typical CNN input sizes. Adjust if necessary.
        transforms.ToTensor(),
    ])

    # Here, the AlzheimerDataset class is instantiated twice, for training and testing purposes. 
    train_dataset = AlzheimerDataset(root_dir, mode='train', transform=transform)
    test_dataset = AlzheimerDataset(root_dir, mode='test', transform=transform)

    # Splitting the training dataset into train and validation subsets
    train_length = int(0.65 * len(train_dataset))
    valid_length = len(train_dataset) - train_length
    train_subset, valid_subset = random_split(train_dataset, [train_length, valid_length])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader