from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader 

# Defaults
BATCH_SIZE = 64
path = "C:\Users\dcp\Documents\OFFLINE-Projects\DATASETS\ADNI"
# path = ...
transforms = None

def load_train_data():
    train_dataset = ImageFolder(path + r"\train")
    return DataLoader(
        train_dataset, 
        batch_size = BATCH_SIZE, 
        transforms = transforms, 
        shuffle = True
    )

def load_test_data():
    test_dataset = ImageFolder(path + r"\test")
    return DataLoader(
        test_dataset, 
        batch_size = BATCH_SIZE, 
        transforms = transforms, 
        shuffle = True
    )
