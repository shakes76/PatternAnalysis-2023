import torch
from torch.utils.data import Dataset, DataLoader
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
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
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
    original_dataset = ADNIDataset(root_dir, transform=transform_original)
    downsampled_dataset = ADNIDataset(root_dir, transform=transform_downsampled)

    # Splitting datasets into training, validation, and testing
    train_size = int(0.8 * len(original_dataset))
    val_size = int(0.1 * len(original_dataset))
    test_size = len(original_dataset) - train_size - val_size

    original_train, original_val, original_test = torch.utils.data.random_split(original_dataset, [train_size, val_size, test_size])
    downsampled_train, downsampled_val, downsampled_test = torch.utils.data.random_split(downsampled_dataset, [train_size, val_size, test_size])

    # Data loaders
    train_loader = DataLoader(list(zip(downsampled_train, original_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(list(zip(downsampled_val, original_val)), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(list(zip(downsampled_test, original_test)), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders("path_to_ADNI_dataset")
    for downsampled, original in train_loader:
        # Example usage
        pass
