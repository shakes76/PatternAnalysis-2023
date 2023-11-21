"""
A custom data set class for the ISIC 2017 data set. This class is used in the train.py and predict.py files.
"""

import numpy as np
import os
import PIL.Image as Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class ISICDataset(Dataset):
    """
    A custom data set class for the ISIC 2017 data set.
    """
    def __init__(self, image_dir, mask_dir, transform=None):
        """Initializes the data set class for the ISIC 2017 data set.

        Args:
            image_dir (str): Drirectory containing the images
            mask_dir (str): Directory containing the masks
            transform (transform, optional): torchvision or albumentations transforms. Defaults to None.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # image list should be all .jpg images in the image_dir, NOT .png
        self.images = [img for img in os.listdir(self.image_dir) if img.endswith('.jpg')]
        # self.image_list = os.listdir(self.image_dir)
        # self.image_list = [os.path.join(self.image_dir, i) for i in self.image_list if i.endswith('.jpg')]

    def __len__(self):
        """Returns the length of the data set

        Returns:
            int: The length of the data set
        """
        return len(self.images)
    
    def __getitem__(self, index):
        """Gets an image and corresponding mask from the data set, applies transforms if specified.

        Args:
            index (int): The index of the image to get

        Returns:
            tuple(torch.Tensor): The image and mask as a tuple of torch tensors
        """
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index][:-4] + '_segmentation.png')
        image = np.array(Image.open(image_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)
        mask[mask == 255.0] = 1.0 # convert all 255 values to 1.0 to make it a binary mask
        
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            image = TF.to_tensor(image)
            mask = TF.to_tensor(mask)
        
        return image, mask

