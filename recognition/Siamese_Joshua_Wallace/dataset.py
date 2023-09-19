import torch
from torchvision import datasets, transforms

# Define transformations
# Note: You might want to use different transforms for training and testing datasets. For simplicity, the same transforms are used here.
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
])

def get_dataloaders(batch_size=32, root_dir='./AD_CN'):
    # Create training and testing datasets
    train_dataset = datasets.ImageFolder(root=f"{root_dir}/train", transform=transform)
    test_dataset = datasets.ImageFolder(root=f"{root_dir}/test", transform=transform)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

