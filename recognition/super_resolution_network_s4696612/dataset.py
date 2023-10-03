from PIL import Image
from torch.utils.data import Dataset
import os


class ImageDataset(Dataset):
    """
    A custom Pytorch Dataset for the training of an
    efficient subpixel CNN for the ADNI brain dataset.
    """
    def __init__(self, directory, transform=None):
        """
        Initialises the dataset with relevant transforms if applicable.
        """
        self.directory = directory
        self.data = os.listdir(directory)
        self.transform = transform
    
    def __len__(self):
        """
        Returns the size of the dataset.
        """
        return len(self.data)
    
    def __getitem__(self, index):
        """
        Returns the downsampled image and the original image at the specified index.
        The downsampled image will be of size 60x64, and the original image
        will be unchanged. For the ADNI dataset, this means the image will be of size
        240x256. This method also applies applicable transformations to the images.
        """
        image_path = os.path.join(self.directory, self.data[index])
        label = Image.open(image_path)
        image = label.resize((64,60))
        if self.transform:
            label = self.transform(label)
            image = self.transform(image)
        return image, label