import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

class SkinDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_segmentation.png"))

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augemntations = self.transform(image=image, mask=mask)
            image = augemntations["image"]
            mask = augemntations["mask"]

        return image, mask

def get_loaders(train_dir, mask_dir, batch_size, train_trasform, val_transform):

    train_dataset = SkinDataset(
        image_dir=train_dir,
        mask_dir=mask_dir,
        transform=train_trasform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return train_loader
    
    