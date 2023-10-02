import torchvision
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def load_train_data():
    path="/home/groups/comp3710/ADNI/AD_NC/train"
    # this for testing
    path="./AD_NC/train"

    # Data transformation
    transform_train = transforms.Compose([
        transforms.Resize(105),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # transform_train = transforms.Compose([
    #     transforms.Resize(105),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.CenterCrop(105),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])

    trainset = torchvision.datasets.ImageFolder(root=path, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

    data_iter = iter(train_loader)
    images, _ = next(data_iter)

    # Check the number of channels in the first image
    num_channels = images.size(1)
    print(f"Number of channels in the first image: {num_channels}")

    return train_loader

def load_test_data():
    path="/home/groups/comp3710/ADNI/AD_NC/test"
    # this for testing
    path="./AD_NC/test"
    # dataset = torchvision.datasets.ImageFolder(root=path, transform=transforms.ToTensor())

    # # Calculate the mean and standard deviation
    # mean = torch.mean(dataset.data.float() / 255.0)
    # std = torch.std(dataset.data.float() / 255.0)

    # Data transformation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # standard scaling for normalize, doesn't know much on the status of the entire dataset
    ])

    testset = torchvision.datasets.ImageFolder(root=path, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=0)
    return test_loader

if __name__ == '__main__':
    # test dataset

    load_train_data()