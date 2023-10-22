import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class CustomDataset(Dataset):
    """
    CustomDataset class for loading images and their corresponding masks from specified directories.

    :param image_dir: Directory containing the images.
    :param mask_dir: Directory containing the corresponding masks.
    :param transform: (Optional) Transformations to be applied on the images and masks.
    """
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Initializes the dataset object with directory paths and transformation.

        :param image_dir: Directory path to the images.
        :param mask_dir: Directory path to the corresponding masks.
        :param transform: (Optional) Transformations to be applied.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        """
        Returns the total number of images in the dataset.
        
        :return: Integer count of images.
        """
        return len(self.images)

    def __getitem__(self, index):
        """
        Fetches and returns the image-mask pair for the given index.

        :param index: Index for the image-mask pair in the dataset.
        
        :return: A tuple containing the image and its corresponding mask.
        """
        
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_segmentation.png"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

