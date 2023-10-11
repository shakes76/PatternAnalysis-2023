from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from math import log2


def load_dataset(image_size, batch_sizes, path):
    # transform
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # data
    batch_size = batch_sizes[int(log2(image_size / 4))]
    dataset = ImageFolder(path, transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    return dataloader, dataset

# KEEP PATIENTS DATA TOGETHER TO PREVENT DATA LEAKAGE AND OVERFITTING