from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader 

def load_train_data(root_dir, batch_size, transforms = None):
    train_dataset = ImageFolder(root_dir + r"\train", transform = transforms)
    return DataLoader(
        train_dataset, 
        batch_size = batch_size, 
        shuffle = True
    )

def load_test_data(root_dir, batch_size, transforms = None):
    test_dataset = ImageFolder(root_dir + r"\test", transform = transforms)
    return DataLoader(
        test_dataset, 
        batch_size = batch_size,
        shuffle = True
    )
