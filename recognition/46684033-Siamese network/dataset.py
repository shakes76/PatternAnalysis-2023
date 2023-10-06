# dataset.py
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

# Path for dataset
train_path = r"C:/Users/wongm/Downloads/ADNI_AD_NC_2D/AD_NC/train"
test_path = r"C:/Users/wongm/Downloads/ADNI_AD_NC_2D/AD_NC/test"
def load_data(train_path,test_path):
    dataset = torchvision.datasets.ImageFolder(root=train_path)
    testset = torchvision.datasets.ImageFolder(root=test_path)

    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    trainset, validation_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=128, shuffle=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128)
    return train_loader, validation_loader, test_loader

