import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, random_split

# Define the dataset root directory and transformation for data preprocessing.
data_root = '/ISIC-2017_Training_Data'
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize the images to 128x128.
    transforms.ToTensor(),  # Convert to a PyTorch tensor.
])

# Load the ISIC 2017 dataset.
def load_isic_dataset(batch_size, split_ratio=(0.8, 0.1, 0.1), num_workers=4):
    dataset = ImageFolder(root=data_root, transform=transform)

    # Split the dataset into training, validation, and test sets.
    train_size = int(len(dataset) * split_ratio[0])
    val_size = int(len(dataset) * split_ratio[1])
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    # Create DataLoader instances for each split.
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
