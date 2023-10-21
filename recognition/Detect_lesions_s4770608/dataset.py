import os
import torch
import torchvision.ops
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision.transforms import functional as F, InterpolationMode

import torch
import numpy as np
from PIL import Image

def mask_to_bbox(mask):
    pos = torch.nonzero(mask, as_tuple=True)
    xmin = torch.min(pos[2])
    xmax = torch.max(pos[2])
    ymin = torch.min(pos[1])
    ymax = torch.max(pos[1])

    # 检查宽度和高度是否为正
    assert (xmax - xmin) > 0 and (ymax - ymin) > 0, f"Invalid bbox: {[xmin, ymin, xmax, ymax]}"

    return torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)


def get_targets_from_mask(mask, label):
    """从mask获取目标字典.
    Args:
    - mask (Tensor): (H, W)大小的二进制掩码图像.
    - label (int): 目标的标签值.
    Returns:
    - target (dict): Mask R-CNN所需的目标格式.
    """
    mask = mask.unsqueeze(0)  # 添加一个额外的batch维度
    bbox = mask_to_bbox(mask)
    target = {
        "boxes": bbox,
        "labels": torch.tensor([label], dtype=torch.int64),
        "masks": mask
    }
    return target



class CustomISICDataset(Dataset):
    def __init__(self, csv_file, img_dir, mask_dir, transform_stage1_for_img_mask=None,transform_stage2_for_img = None, target_size=224):
        self.labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform_stage1_for_img_mask = transform_stage1_for_img_mask
        self.transform_stage2_for_img=transform_stage2_for_img
        self._check_dataset_integrity()

    def _check_dataset_integrity(self):
        # Getting image and mask file names without extensions
        image_ids = self.labels['image_id'].tolist()
        mask_ids = [
            os.path.splitext(mask_file)[0].replace('_segmentation', '')
            for mask_file in os.listdir(self.mask_dir)
        ]

        # Checking if lengths are the same
        assert len(image_ids) == len(mask_ids), \
            f"Number of images ({len(image_ids)}) and masks ({len(mask_ids)}) do not match."

        # Checking if filenames correspond
        assert set(image_ids) == set(mask_ids), \
            "Image IDs and Mask IDs do not correspond."

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = self.labels.iloc[idx, 0]
        img_path = f"{self.img_dir}/{img_name}.jpg"
        mask_path = f"{self.mask_dir}/{img_name}_segmentation.png"

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # assuming mask is 1 channel
        if self.transform_stage1_for_img_mask:
            image,mask = self.transform_stage1_for_img_mask([image,mask])
        count = 0
        while True:
            if mask.sum()>100 :
                break
            if count >10:
                return  None
            else:
                count+=1
                image, mask = self.transform_stage1_for_img_mask([image, mask])

        if self.transform_stage2_for_img:
            image,mask = self.transform_stage2_for_img([image,mask])
        # Your class labels
        melanoma = int(self.labels.iloc[idx, 1])
        seborrheic_keratosis = int(self.labels.iloc[idx, 2])

        # Define label based on your conditions
        if melanoma == 1 and seborrheic_keratosis == 0:
            label = 1
        elif melanoma == 0 and seborrheic_keratosis == 1:
            label = 2
        elif melanoma == 0 and seborrheic_keratosis == 0:
            label = 3
        else:
            raise ValueError("Invalid label found!")
        # Resize image and mask
        bbox = torchvision.ops.masks_to_boxes(mask)
        target = {
            "boxes": bbox,
            "labels": torch.tensor([label], dtype=torch.int64),
            "masks": mask
        }



        return image, target
