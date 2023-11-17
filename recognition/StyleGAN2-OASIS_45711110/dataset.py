'''
The module reads and loads data.
The data is augmented and transformed during import for faster training.
'''

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

from config import channels

'''
Data Loader

Resize: Resize images to the specific resolution
RandomHorizontalFlip: Augment data by applying random horizontal flips [probability=50%]
ToTensor: Convert images to PyTorch Tensors
Normalize: Normalize pixel value to have a mean and standard deviation of 0.5
'''
def get_data(data, log_res, batchSize):

    if channels == 3:
        transform = v2.Compose([
                v2.ToTensor(),
                v2.RandomVerticalFlip(p=0.5),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
            )
    elif channels == 1:
        transform = v2.Compose([
                v2.ToTensor(),
                v2.Grayscale(),
                v2.RandomVerticalFlip(p=0.5),
                v2.Normalize(mean=[0.5], std=[0.5])]
            )

    dataset = datasets.ImageFolder(root=data, transform=transform)

    loader = DataLoader(dataset, batchSize, shuffle=True)

    return loader
