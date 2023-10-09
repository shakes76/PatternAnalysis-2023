import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import os

class ADNIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(root_dir) for f in filenames if f.endswith('.jpeg')]  # Assuming images are in .jpeg format

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image

def downsample_transform(factor=4):
    return transforms.Compose([
        Resize((256 // factor, 240 // factor)),  # Assuming images are 256x240.
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Assuming Min-Max normalization to [0,1]. Adjust if using Z-score normalization.
    ])

def original_transform():
    return transforms.Compose([
        Resize((256, 240)),  # Resize to a standard size
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Assuming Min-Max normalization to [0,1]. Adjust if using Z-score normalization.
    ])

def get_dataloaders(root_dir, batch_size=32):
    # Transforms
    transform_original = original_transform()
    transform_downsampled = downsample_transform()

    # Datasets
    original_train_dataset = ADNIDataset(os.path.join(root_dir, 'train'), transform=transform_original)
    downsampled_train_dataset = ADNIDataset(os.path.join(root_dir, 'train'), transform=transform_downsampled)
    
    original_test_dataset = ADNIDataset(os.path.join(root_dir, 'test'), transform=transform_original)
    downsampled_test_dataset = ADNIDataset(os.path.join(root_dir, 'test'), transform=transform_downsampled)

    # Data loaders
    train_loader = DataLoader(list(zip(downsampled_train_dataset, original_train_dataset)), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(list(zip(downsampled_test_dataset, original_test_dataset)), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
