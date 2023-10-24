from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_dataloaders(train_string, test_validation_string, batch_size):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5], std=[0.5])  
    ])
    train_dataset = datasets.ImageFolder(root='train_string', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    full_test_dataset = datasets.ImageFolder(root='test_validation_string', transform=transform)
    test_size = int(0.3 * len(full_test_dataset))
    val_size = len(full_test_dataset) - test_size
    
    test_dataset, val_dataset = random_split(full_test_dataset, [test_size, val_size])
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
