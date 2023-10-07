from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader



image_size = 64
batch_size = 128

# transform
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# data
dataset = ImageFolder('~/data/celeba', transform=transform)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)   