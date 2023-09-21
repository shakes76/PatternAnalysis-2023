from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def train_transforms():
    """
    Transforms to be applied to the training dataset
    """
    return transforms.Compose([transforms.Normalize(0.1155, 0.2254), transforms.Grayscale(), transforms.ToTensor()])

def test_transforms():
    """
    Transforms to be applied to the test dataset
    """
    return transforms.Compose([transforms.Normalize(0.1155, 0.2254), transforms.Grayscale(), transforms.ToTensor()])

def train_dataloader(dir, batch_size):
    train_dataset = datasets.ImageFolder(root = dir + "/train", transform=train_transforms())
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader
    
def test_dataloader(dir, batch_size):
    test_dataset = datasets.ImageFolder(root = dir + "/test", transform=test_transforms())
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    return test_dataloader