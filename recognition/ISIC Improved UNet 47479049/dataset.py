import os
import torch
import torch.utils.data
import torchvision.transforms as T
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

        mask[mask <= 128.0] = 0
        mask[mask > 128.0] = 1

        image = self.transform(image)
        mask = self.transform(mask)

        return image, mask
    

    
class TestDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.image_dir = image_dir
        self.images = os.listdir(image_dir)
        self.transfrom = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))

        t_image = self.transfrom(image)

        return t_image
        

def get_loaders(train_dir, mask_dir, test_dir, batch_size, train_trasform, val_transform):

    dataset = SkinDataset(
        image_dir=train_dir,
        mask_dir=mask_dir,
        transform=train_trasform
    )

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(1))

    test_dataset = TestDataset(
        image_dir=test_dir,
        transform=train_trasform
    )

    test_loader = DataLoader(
        test_dataset, 
        num_workers=8
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=8
    )


    return train_loader, val_loader, test_loader
    
    