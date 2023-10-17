import torch
import torchvision.transforms as transforms
from PIL import Image
import os

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, imageTransform=None, maskTransform=None):
        prefix = "ISIC"
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.imageTransform = imageTransform
        self.maskTransform = maskTransform
        self.images = os.listdir(image_dir)
        self.images = [image for image in self.images if image.startswith(prefix)]
        self.masks = os.listdir(mask_dir)
        self.masks = [mask for mask in self.masks if mask.startswith(prefix)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        mask = Image.open(mask_path).convert('L')
        if self.imageTransform:
            image = self.imageTransform(image)
        if self.maskTransform:
            mask = self.maskTransform(mask)
        return (image, mask)
