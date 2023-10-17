from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader 

def load_train_data(root_dir, batch_size, transforms = None):
    train_dataset = ImageFolder(root_dir + r"\train")
    return DataLoader(
        train_dataset, 
        batch_size = batch_size, 
        transforms = transforms, 
        shuffle = True
    )

def load_test_data(root_dir, batch_size, transforms = None):
    test_dataset = ImageFolder(root_dir + r"\test")
    return DataLoader(
        test_dataset, 
        batch_size = batch_size, 
        transforms = transforms, 
        shuffle = True
    )
