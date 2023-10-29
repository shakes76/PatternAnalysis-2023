'''
The module reads and loads data.
The data is augmented and transformed during import for faster training.
A sample image of the data after loading is shown

v2: Uses 1 channel since img is grayscale
'''

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

'''
Data Loader

Resize: Resize images to the specific resolution
RandomHorizontalFlip: Augment data by applying random horizontal flips [probability=50%]
ToTensor: Convert images to PyTorch Tensors
GrayScale: Since the images are black&white, the data is transformed to use 1 channel only
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

    # print an image from the loaded dataset
    sample_img(loader)

    return loader

# Prints a tensor shape and a random image from loader in grayscale
def sample_img(loader):
    features, _ = next(iter(loader))
    print(f"Feature batch shape: {features.size()}")
    img = features[0].squeeze()
    plt.imshow(img, cmap="gray")
    plt.show()