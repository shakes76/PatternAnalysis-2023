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

        # Changes image to numpy array then changes to values to rbg(3 channels) L(greysacle 1 channel) 
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)


        # if pixel is black change to 1
        mask[mask <= 128.0] = 0
        mask[mask > 128.0] = 1

        image = self.transform(image)
        mask = self.transform(mask)

        return image, mask

def get_loaders(train_dir, mask_dir, batch_size, train_trasform):
    """
    get_loaders takes in the directory of the data set then converts them into dataloaders
    train_dir: directory of training dataset
    mask_dir: directory of mask dataset
    batch_size: batch size for loaders
    trian_transform: transormation for dataset
    
    returns: Training Dataloder and validation Dataloader
    """

    dataset = SkinDataset(
        image_dir=train_dir,
        mask_dir=mask_dir,
        transform=train_trasform
    )

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(1))
    
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


    return train_loader, val_loader
    
    