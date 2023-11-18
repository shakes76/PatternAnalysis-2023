'''
The module reads and loads data.
The data is augmented and transformed during import for faster training.
'''

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from config import channels, image_height, image_width

'''
Saves 5 images after the data transformation/augmentation and loading is complete and wrapped using dataloader.
'''
def show_imgs(loader):    
    
    for i in range(5):
        features, _ = next(iter(loader))
        print(f"Feature batch shape: {features.size()}")
        img = features[0].squeeze()
        plt.imshow(img, cmap="gray")
        save_image(img*0.5+0.5, f"aug_img_{i}.png")

'''
Data Loader

Resize: Resize images to a lower resolution as set in config, uses bicubic interpolation
RandomHorizontalFlip: Augment data by applying random horizontal flips [probability=50%]
ToTensor: Convert images to PyTorch Tensors
Normalize: Normalize pixel value to have a mean and standard deviation of 0.5
'''
def get_data(data, log_res, batchSize):

    if channels == 3:
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((image_height, image_width), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
            )
    elif channels == 1:
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((image_height, image_width), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.Grayscale(),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.Normalize(mean=[0.5], std=[0.5])]
            )

    dataset = datasets.ImageFolder(root=data, transform=transform)

    loader = DataLoader(dataset, batchSize, shuffle=True)

    if channels == 1:
        show_imgs(loader)
        
    return loader
