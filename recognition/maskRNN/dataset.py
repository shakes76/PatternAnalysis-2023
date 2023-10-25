import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import os
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    return T.Compose(transforms)

class isicData(Dataset):
    def __init__(self, images, masks, diagnoses):
        self.images = images
        self.masks = masks
        self.diagnoses = pd.read_csv(diagnoses)
        self.len = self.diagnoses.shape[0]
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.transform = get_transform(True)

    def get_path(self, is_mask, id):
       path = self.images
       extension = ".jpg"
       if is_mask:
           path = self.masks
           extension = "_segmentation.png"

       return os.path.join(path, id + extension)
    
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        record = self.diagnoses.iloc[idx]
        id = record["image_id"]
        label_value = 1 if record["melanoma"] else 0

        imagePath = self.get_path(is_mask=False, id=id)
        maskPath = self.get_path(is_mask=True, id=id)
        img_data = Image.open(imagePath).convert("RGB")
        mask_data = Image.open(maskPath)
        
        mask_array = np.array(mask_data)
        unique_obj_values = np.unique(mask_array)[1:]

        boolean_masks = mask_array == unique_obj_values[:, None, None]
        num_objects = len(unique_obj_values)

        bounding_boxes = []
        for obj_index in range(num_objects):
            coords = np.where(boolean_masks[obj_index])
            left, right = np.min(coords[1]), np.max(coords[1])
            top, bottom = np.min(coords[0]), np.max(coords[0])
            bounding_boxes.append([left, top, right, bottom])

        bounding_boxes = torch.tensor(bounding_boxes, dtype=torch.float32)
        label_tensor = torch.tensor([label_value], dtype=torch.int64)
        mask_tensor = torch.tensor(boolean_masks, dtype=torch.uint8)

        box_area = (bounding_boxes[:, 3] - bounding_boxes[:, 1]) * (bounding_boxes[:, 2] - bounding_boxes[:, 0])
        crowd_status = torch.zeros((1,), dtype=torch.int64)

        object_data = {
        "boxes": bounding_boxes,
        "labels": label_tensor,
        "masks": mask_tensor,
        "area": box_area,
        "iscrowd": crowd_status
        }
        img_data = self.transform(img_data)
        return img_data, object_data
    


            
