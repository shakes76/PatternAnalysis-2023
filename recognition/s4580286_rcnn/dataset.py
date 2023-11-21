'''
Loads the data from the file directory. 
Outputs: image(tensor), targets(dict((tensor), (tensor), (tensor))

'''
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from torchvision.ops import masks_to_boxes

class MoleData(Dataset):
    def __init__(self, image_path, mask_path, diag_path):
        '''
        Takes inputs of image_path, diagnostic_path(.csv) and mask_paths

        '''
        self.image_path = image_path
        self.mask_path = mask_path
        self.diagnosis = pd.read_csv(diag_path)


    def __getitem__(self, index):
        '''
        Load and process single sample
        Outputs : images, targets

        '''
        # Get Index
        image_idx = self.diagnosis.iloc[index]["image_id"]
        melanoma_status = self.diagnosis.iloc[index]["melanoma"]

        #Specify Label
        if melanoma_status:
            has_melanoma = 2 #Has melanoma
        else:
            has_melanoma = 1

        #Transform image
        im_path = os.path.join(self.image_path, image_idx + ".jpg")
        image = Image.open(im_path)
        image = image.convert("RGB")

        image = transforms.PILToTensor()(image)
        image = transforms.ConvertImageDtype(torch.float)(image)
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(image)

        #Get mask
        mask_path = os.path.join(self.mask_path, image_idx + "_segmentation.png")
        mask = read_image(mask_path)
        if mask is not None:
          obj_ids = torch.unique(mask)
          obj_ids = obj_ids[1:]
          masks = mask == obj_ids[:, None, None]
          boxes = masks_to_boxes(masks)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor([has_melanoma], dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        targets = {}
        targets["labels"] = labels
        targets["boxes"] = boxes
        targets["masks"] = masks
        return image, targets


