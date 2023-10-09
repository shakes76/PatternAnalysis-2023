import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Define data transformation
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Apply transformations to datasets
train_dataset = ImageFolder(root='/Users/mj/Documents/COMP3710_code/COMP3710_Report/AD_NC/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = ImageFolder(root='/Users/mj/Documents/COMP3710_code/COMP3710_Report/AD_NC/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

train_img, train_label = next(iter(train_loader))
print(train_img.shape)
print(train_label)

test_img, test_label = next(iter(test_loader))
print(test_img.shape)
print(test_label)


