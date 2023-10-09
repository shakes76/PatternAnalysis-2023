import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
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
        self.image_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(root_dir) for f in filenames if f.endswith('.png')]  # Assuming images are in .png format

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image

def downsample_transform(factor=4):
    return Compose([
        Resize((256 // factor, 240 // factor)),  # Assuming images are 256x256. Adjust if different.
        ToTensor()
    ])

def original_transform():
    return Compose([
        Resize((256, 240)),  # Resize to a standard size
        ToTensor()
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

train_loader, test_loader = get_dataloaders("C:\\Users\\soonw\\ADNI\\AD_NC")
    
