import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dataset: {torch.cuda.is_available()}")

class ISICDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, img_transform=None, mask_transform=None, img_size=None):
        self.img_size = img_size
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.img_transform = img_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        image = image.resize(self.img_size, Image.BILINEAR)

        mask = None  # We start by assuming there is no mask
        if self.mask_dir:  # If a mask directory is provided
            mask_name = self.images[idx].replace('.jpg', '_segmentation.png')
            mask_path = os.path.join(self.mask_dir, mask_name)

            if os.path.exists(mask_path):  # Check if the mask file exists
                mask = Image.open(mask_path).convert("L")
                mask = mask.resize(self.img_size, Image.NEAREST)
            else:
                print(f"File not found: {mask_path}")

                # Apply transformations if specified
        if self.img_transform:
            image = self.img_transform(image)

        if mask is not None and self.mask_transform:  # Only apply mask transform if mask is present
            mask = self.mask_transform(mask)

        return image, mask

def get_mask_transform():
    # Masks usually don't require normalization, but they still need to be tensors
    return transforms.Compose([
        transforms.ToTensor()
    ])


# Define the transformation function
def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

