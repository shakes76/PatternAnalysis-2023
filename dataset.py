from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
dataset_train = datasets.ImageFolder('../ADNI_AD_NC_2D/AD_NC/train/', transform=transform)
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
dataset_test = datasets.ImageFolder('../ADNI_AD_NC_2D/AD_NC/test/', transform=transform)
dataset_test, dataset_val = random_split(dataset_test, [0.7, 0.3])
dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=True)

def returnDataloaders():
    return dataloader_train, dataloader_test, dataloader_val