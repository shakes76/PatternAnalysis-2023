import os
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import InterpolationMode
import numpy as np
import random

# Loading the data (source: https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet/dataset.py)
class ISICDataLoader(Dataset):
    def __init__(self, image_dir, mask_dir, split=(0.8, 0.1, 0.1), transform=None):
        """
        Initializes the ISICDataLoader with the specified parameters and splits the dataset.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.split = split

        self.split_dataset()

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.images)


    def __getitem__(self, index):
        """
        Loads and returns an image and its corresponding mask at the given index.
        
        Args: index (int): The index of the sample to retrieve.
        
        Returns: tuple: A tuple containing the loaded image and its corresponding mask.
        """
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".png", "_mask.gif"))

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()]
    )

    def split_dataset(self):
        """
        Splits the dataset into training, validation, and test sets based on the provided split ratios.
        """
        random.shuffle(self.images)
        total_samples = len(self.images)
        split_idx = [int(total_samples * self.split[0]),
                     int(total_samples * (self.split[0] + self.split[1]))]

        train_images = self.images[:split_idx[0]]
        val_images = self.images[split_idx[0]:split_idx[1]]
        test_images = self.images[split_idx[1]:]

        self.dataset = [(img, 0) for img in train_images] + [(img, 1) for img in val_images] + [(img, 2) for img in test_images]

        




