import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.image_files = [file for file in os.listdir(image_dir) if file.endswith('.jpg')]
        self.mask_files = [file for file in os.listdir(mask_dir) if file.endswith('_segmentation.png')]

        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        mask_name = os.path.join(self.mask_dir, self.mask_files[idx])

        image = Image.open(img_name)
        mask = Image.open(mask_name)

        # Ensure consistent sizing
        image = image.resize((256, 256), Image.ANTIALIAS)
        mask = mask.resize((256, 256), Image.ANTIALIAS)

        if self.transform is not None:
            image = self.transform(image)

        # Convert the grayscale mask to a binary mask
        mask = self.normalize_mask(mask)

        return image, mask

    def normalize_mask(self, mask):
        #normalise mask to a Pytorch tensor
        mask = np.array(mask)
        mask = mask / 255.0
        mask = torch.tensor(mask, dtype=torch.float32)
        return mask

def get_loader(image_dir, mask_dir, batch_size, transform=None):
    if transform is None:
        transform = transforms.Compose([transforms.ToTensor()])

    dataset = CustomDataset(image_dir, mask_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader
