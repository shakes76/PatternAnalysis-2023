from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader



image_size = 64
batch_size = 128
path = '~/data/celeba'


def load_dataset(image_size, batch_size, path):
    # transform
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # data
    dataset = ImageFolder(path, transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    return dataloader

dataloader = load_dataset(image_size, batch_size, path)