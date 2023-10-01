import torchvision
import torch
import torchvision.transforms as transforms


def load_train_data():
    dataset = torchvision.datasets.ImageFolder(root="/home/groups/comp3710/ADNI/train", transform=transforms.ToTensor())

    # Calculate the mean and standard deviation
    mean = torch.mean(dataset.data.float() / 255.0)
    std = torch.std(dataset.data.float() / 255.0)

    # Data transformation
    transform_train = transforms.Compose([
        transforms.Resize(128),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean, std) # standard scaling for normalize, doesn't know much on the status of the entire dataset
    ])

    trainset = torchvision.datasets.ImageFolder(root="/home/groups/comp3710/ADNI/train", transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

    return train_loader

def load_test_data():

    dataset = torchvision.datasets.ImageFolder(root="/home/groups/comp3710/ADNI/train", transform=transforms.ToTensor())

    # Calculate the mean and standard deviation
    mean = torch.mean(dataset.data.float() / 255.0)
    std = torch.std(dataset.data.float() / 255.0)

    # Data transformation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std) # standard scaling for normalize, doesn't know much on the status of the entire dataset
    ])

    testset = torchvision.datasets.ImageFolder(root="/home/groups/comp3710/ADNI/test", transform=transform_test)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=0)
    return test_loader