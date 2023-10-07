import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as T
import numpy as np


def get_transform():
    return T.Compose([
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float)
    ])


class ISICDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.image_files[idx].replace('.jpg', '_segmentation.png'))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            mask = mask.to(dtype=torch.int64)

        # Compute bounding boxes from masks
        pos = np.where(np.array(mask) > 0)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        boxes = torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)

        # Create target dictionary
        target = {
            "boxes": boxes,
            "labels": torch.ones((1,), dtype=torch.int64), 
            "masks": mask,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((1,), dtype=torch.int64)
        }

        return image, target


if __name__ == "__main__":
    TRAIN_IMG_DIR = './ISIC2018_Task1-2_Training_Input'  
    TRAIN_MASK_DIR = './ISIC2018_Task1_Training_GroundTruth'  

    train_dataset = ISICDataset(img_dir=TRAIN_IMG_DIR, mask_dir=TRAIN_MASK_DIR, transform=get_transform())
