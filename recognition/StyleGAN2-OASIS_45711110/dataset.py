'''
The module reads and loads data.
The data is augmented and transformed during import for faster training.
'''

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

'''
Data Loader

Resize: Resize images to the specific resolution
RandomHorizontalFlip: Augment data by applying random horizontal flips [probability=50%]
ToTensor: Convert images to PyTorch Tensors
Normalize: Normalize pixel value to have a mean and standard deviation of 0.5
'''
def get_data(data, log_res, batchSize):

    transform = transforms.Compose(
        [   transforms.Resize(size=(2**log_res, 2**log_res), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(mean=[0.5], std=[0.5])
            ]
        )

    dataset = datasets.ImageFolder(root=data, transform=transform)

    loader = DataLoader(dataset, batchSize, shuffle=True)

    return loader