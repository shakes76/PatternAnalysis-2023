from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
from os.path import join as pathJoin
from PIL import Image
from torchvision.transforms import v2
from CONFIG import *

class ISICDataloader(Dataset):
    """
    Creates the Dataloader from the ISIC dataset

    Parameters:
        classify_file (string): Path to the csv file containning image names and classification
        photo_dir (string): Path to original images
        mask_dir (string): Path to the segmentation masks
        mask_empty_dim (tuple): HxW tuple for empty masks, (deprecanted)
        S (int): The amount of SxS cells to split the data into, strongly recommended don't change
        B (int): The amount of bounding boxes to predict in each cell for multiple objects
        C (int): The number of classes, i.e. melanoma and seborrheic_keratosis
    """
    def __init__(self, classify_file, photo_dir, mask_dir, mask_empty_dim=image_size, S=1, B=1, C=2, transform=None) -> None:
        self.device = self.check_cuda()
        self.csv_df = pd.read_csv(classify_file)
        self.photo_dir = photo_dir
        self.mask_dir = mask_dir
        self.length = self.csv_df.shape[0]
        self.empty_H = mask_empty_dim[1]
        self.empty_W = mask_empty_dim[0]
        self.S = S
        self.B = B
        self.C = C
        self.transform = transform
        self.defaultTransform = v2.Compose([
            v2.ToImage(), 
            v2.ToDtype(torch.float32),
            v2.Resize(size=image_size, antialias=True)
        ])
        self.class_dictionary = {0: 'melanoma', 1: 'seborrheic_keratosis'}

    """
    Open cuda, do not run at all if can't use gpu
    """
    def check_cuda(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            exit("Warning CUDA not Found. Using CPU")
        return device
    
    """
    Turn a mask (HxW) into a bounding box of x1, y1, x2, y2
    """
    def mask_to_bbox(self, mask):
        non_zero_coords = torch.nonzero(mask)
        min_coords = non_zero_coords.min(axis=0)
        max_coords = non_zero_coords.max(axis=0)
        return (min_coords.values[1].item(), min_coords.values[0].item(), 
                max_coords.values[1].item(), max_coords.values[0].item())
    
    """
    Used to turn a bounding box of x1, y1, x2, y2 to a centred bounding box
    of x, y, w, h. To be used with mask_to_bbox.
    """
    def bbox_to_XYWH(self, bbox):
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]
        x = (x1 + x2) / 2 / image_size[1]
        y = (y1 + y2) / 2 / image_size[0]
        w = abs(x1 - x2) / image_size[1]
        h = abs(y1 - y2) / image_size[0]
        return x, y, w, h

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        # Need boxes, labels and masks
        row = self.csv_df.iloc[index]
        mask_pth = pathJoin(self.mask_dir, row['image_id'])
        img_pth = pathJoin(self.photo_dir, row['image_id'])
        #print(row['image_id'])
        class_label = -1
        # Find which class label for object
        for key, value in self.class_dictionary.items():
            if row[value] == 1:
                class_label = key
                break

        # Find bounding boxes or empty if none
        if class_label >= 0:
            masks = Image.open(mask_pth + '_segmentation.png').convert('L')
            masks = self.defaultTransform(masks).to(self.device) / 255.0
            bboxes = torch.tensor([class_label, *self.bbox_to_XYWH(self.mask_to_bbox(masks[0]))]).unsqueeze(0)
        else:
            bboxes = torch.empty(0, 5)
        
        image = Image.open(img_pth + '.jpg').convert('RGB')
        image = self.defaultTransform(image).to(self.device) / 255.0

        if self.transform:
            image = self.transform(image)
        
        # Find bounding box for each cell
        label_matrix = torch.zeros((self.S, self.S, (self.C + 5) * self.B))
        for box in bboxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            """
            Calculating the width and height of cell of bounding box,
            relative to the cell is done by the following, with
            width as the example:
            
            width_pixels = (width*self.image_width)
            cell_pixels = (self.image_width)
            
            Then to find the width relative to the cell is simply:
            width_pixels/cell_pixels, simplification leads to the
            formulas below.
            """
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object
            # per cell!
            # print(i, j)
            if label_matrix[i, j, self.C] == 0:
                # Set that there exists an object
                label_matrix[i, j, self.C] = 1

                # Box coordinates
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 3:7] = box_coordinates

                # Set one hot encoding for class_label
                label_matrix[i, j, class_label] = 1

        return image, label_matrix