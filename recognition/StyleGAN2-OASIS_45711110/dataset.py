'''Data Loader for loading and preprocessing the OASIS data'''

import torch
import torchvision
import torchvision.transforms as transforms

'''
Data Loader

Resize: Resize images to the specific resolution
RandomHorizontalFlip: Augment data by applying random horizontal flips [probability=50%]
ToTensor: Convert images to PyTorch Tensors
Normalize: Normalize pixel value to have a mean and standard deviation of 0.5 (for each channels)
'''

def get_data(dataset, imageHeight, imageWidth, batchSize):
    trainSet = torchvision.datasets.ImageFolder(root=dataset+"train",
                                                transform=transforms.Compose([
                                                    transforms.Resize((imageHeight, imageWidth)),
                                                    transforms.RandomHorizontalFlip(p=0.5),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                ]))
    
    validateSet = torchvision.datasets.ImageFolder(root=dataset+"validate",
                                                transform=transforms.Compose([
                                                    transforms.Resize((imageHeight, imageWidth)),
                                                    transforms.RandomHorizontalFlip(p=0.5),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                ]))
    
    testSet = torchvision.datasets.ImageFolder(root=dataset+"test",
                                                transform=transforms.Compose([
                                                    transforms.Resize((imageHeight, imageWidth)),
                                                    transforms.RandomHorizontalFlip(p=0.5),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                ]))

    # Data Loader: Forms batch and shuffles the data
    train_loader = torch.utils.data.DataLoader(trainSet, batch_size=batchSize, shuffle=True)  
    validate_loader = torch.utils.data.DataLoader(validateSet, batch_size=batchSize, shuffle=True) 
    test_loader = torch.utils.data.DataLoader(testSet, batch_size=batchSize, shuffle=True)

    return trainSet, train_loader, testSet, test_loader, validateSet, validate_loader