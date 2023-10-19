from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
from os.path import join as pathJoin
from PIL import Image
from torchvision.transforms import v2

class ISICDataloader(Dataset):
    def __init__(self, classify_file, photo_dir, mask_dir, mask_empty_dim, transform=None) -> None:
        self.device = self.check_cuda()
        self.csv_df = pd.read_csv(classify_file)
        self.photo_dir = photo_dir
        self.mask_dir = mask_dir
        self.length = self.csv_df.shape[0]
        self.empty_H = mask_empty_dim[0]
        self.empty_W = mask_empty_dim[1]
        self.index = 0
        self.transform = transform
        self.defaultTransform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32)])

    def check_cuda(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            exit("Warning CUDA not Found. Using CPU")
        return device
    
    def mask_to_bbox(self, mask):
        non_zero_coords = torch.nonzero(mask)
        min_coords = non_zero_coords.min(axis=0)
        max_coords = non_zero_coords.max(axis=0)
        return torch.tensor([min_coords.values[1].item(), min_coords.values[0].item(), 
                max_coords.values[1].item(), max_coords.values[0].item()], dtype=torch.float32)

    def __len__(self):
        return self.length
    
    def _empty_bbox(self):
        return torch.empty((0, 4), dtype=torch.float32)
    
    def _empty_labels(self):
        return torch.empty(0, dtype=torch.int64)

    def _empty_masks(self):
        return torch.empty((0, self.empty_H, self.empty_W), dtype=torch.uint8)    
    
    def __getitem__(self, index):
        # Need boxes, labels and masks
        row = self.csv_df.iloc[self.index]
        mask_pth = pathJoin(self.mask_dir, row['image_id'])
        img_pth = pathJoin(self.photo_dir, row['image_id'])
        print(row['image_id'])
        if (row['melanoma'] == 1):
            labels = torch.tensor([1], dtype=torch.int64)
            mask = Image.open(mask_pth + '_segmentation.png').convert('L')
            mask = self.defaultTransform(mask).to(self.device) / 255.0
            if self.transform:
                mask = self.transform(mask)
            masks = mask
            boxes = self.mask_to_bbox(masks[0]).unsqueeze(0)
        elif (row['seborrheic_keratosis'] == 1):
            labels = torch.tensor([2], dtype=torch.int64)
            mask = Image.open(mask_pth + '_segmentation.png').convert('L')
            mask = self.defaultTransform(mask).to(self.device) / 255.0
            if self.transform:
                mask = self.transform(mask)
            masks = mask
            boxes = self.mask_to_bbox(masks[0]).unsqueeze(0)
        else: # No images in image
            labels = self._empty_labels()
            masks = self._empty_masks()
            boxes = self._empty_bbox()
        
        self.index += 1
        targets = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks
        }
        image = Image.open(img_pth + '.jpg').convert('RGB')
        image = mask = self.defaultTransform(image).to(self.device) / 255.0
        if self.transform:
            image = self.transform(image)

        return image, targets

        
    