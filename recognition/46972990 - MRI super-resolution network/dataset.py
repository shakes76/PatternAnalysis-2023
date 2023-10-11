import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Paths to the images
BASE_PATH = "C:\\Users\\User\\OneDrive\\Bachelor of Computer Science\\Semester 6 2023\\COMP3710\\ADNI_AD_NC_2D\\AD_NC"
TRAIN_PATH = os.path.join(BASE_PATH, "train")
TEST_PATH = os.path.join(BASE_PATH, "test")

# Image size and batch size
IMG_WIDTH = 256
IMG_HEIGHT = 240
BATCH_SIZE = 32

def get_data_loader(path):
    """
    Given a path (either to train or test), this function will return a DataLoader for the images
    along with some basic transformations.
    """
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizing to [-1, 1]
    ])

    dataset = datasets.ImageFolder(root=path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    return data_loader

# Testing to ensure that the data loader is working correctly
train_loader = get_data_loader(TRAIN_PATH)
dataiter = iter(train_loader)
images, _ = next(dataiter)

print("Images shape:", images.shape)
print(len(train_loader.dataset))
image = images[0].numpy().transpose((1, 2, 0)) * 0.5 + 0.5
plt.imshow(image)
plt.show()