"""
File containing the data loaders used for loading and preprocessing the data.
"""

import os
import torch
from utils import RandomCenterCrop, RandomRotate90, DictTransform
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np

image_path = "/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Training_Input_x2"
mask_path = "/home/groups/comp3710/ISIC2018/ISIC2018_Task1_Training_GroundTruth_x2"
inconsistent_path = "/home/Student/s4745275/PatternAnalysis-2023/recognition/Problem_47452752/inconsistent_ids.txt"

def check_consistency(image_dir=image_path, mask_dir=mask_path, inconsistent_path=inconsistent_path):
    image_ids = {
        img.split(".")[0] for img in os.listdir(image_dir) if img.endswith(".jpg")
    }
    mask_ids = {
        mask.split("_segmentation.")[0]
        for mask in os.listdir(mask_dir)
        if mask.endswith("_segmentation.png")
    }

    # Using list differences to find inconsistencies
    images_without_masks = image_ids - mask_ids
    masks_without_images = mask_ids - image_ids

    if images_without_masks or masks_without_images:
        inconsistent_ids = images_without_masks.union(masks_without_images)
        # Save to a file
        with open(inconsistent_path, "w") as file:
            for ID in inconsistent_ids:
                file.write(f"{ID}\n")

        print(f"Detected {len(inconsistent_ids)} inconsistencies, fixed em tho")


class ISICDataset(Dataset):
    def __init__(
        self,
        transform,
        image_dir=image_path,
        mask_dir=mask_path,
        inconsistent_path=inconsistent_path
    ):
        # Load the inconsistent IDs
        with open(inconsistent_path, "r") as file:
            excluded_ids = set(line.strip() for line in file)

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_ids = [
            img.split(".")[0]
            for img in os.listdir(image_dir)
            if img.endswith(".jpg") and img.split(".")[0] not in excluded_ids
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def handle_inconsistency(self):
        images_without_masks, masks_without_images = check_consistency(
            self.image_dir, self.mask_dir
        )
        inconsistent_ids = images_without_masks.union(masks_without_images)

        # Save to a file
        with open(inconsistent_path, "a") as file:  # 'a' mode for appending
            for ID in inconsistent_ids:
                file.write(f"{ID}\n")

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_ids[idx] + ".jpg")
        mask_name = os.path.join(
            self.mask_dir, self.image_ids[idx] + "_segmentation.png"
        )

        try:
            with Image.open(img_name) as image, Image.open(mask_name) as mask:
                image = image.convert("RGB")
                mask = mask.convert("L")
                sample = {"image": image, "mask": mask}

                if self.transform:
                    sample = self.transform(sample)
                
            # Convert mask to binary 0/1 tensor
            sample["mask"] = (torch.tensor(np.array(sample["mask"])) > 0.5).float()

            return sample["image"], sample["mask"]

        except FileNotFoundError:
            self.handle_inconsistency()
            return self.__getitem__(idx)


# Transformation pipeline to augment the dataset
transform = transforms.Compose(
    [
        RandomRotate90(),
        RandomCenterCrop(),
        DictTransform(transforms.RandomHorizontalFlip()),
        DictTransform(transforms.RandomVerticalFlip()),
        DictTransform(transforms.Resize((256, 256))),
        DictTransform(transforms.ToTensor()),
        DictTransform(
            transforms.Lambda(
                lambda img_tensor: torch.cat(
                    [
                        img_tensor,
                        TF.to_tensor(TF.to_pil_image(img_tensor).convert("HSV")),
                    ],
                    dim=0,
                )
            ),
            False,
        ),
        DictTransform(
            transforms.Normalize(
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
            ),
            False,
        ),
    ]
)
