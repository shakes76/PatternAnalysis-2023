'''
Loads the data from the file directory and processes to output

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
        self.transform = get_transform()

    def __getitem__(self, index):
        '''
        Load and process single sample
        Outputs : has_melanoma

        '''
        # Get Index 
        image_idx = self.diagnosis.iloc[index]["image_id"]
        melanoma_status = self.diagnosis.iloc[index]["melanoma"]

        #Specify Label
        if melanoma_status:
            has_melonoma = 1 #Has melanoma
        else:
            has_melonoma = 0

        #Transform image 
        im_path = os.path.join(self.image_path, image_idx + ".jpg")
        image = Image.open(im_path)
        image = image.convert("RGB")

        image = transforms.PILToTensor()(image)
        image = transforms.ConvertImageDtype(torch.float)(image)

        #Get mask 
        mask_path = os.path.join(self.mask_path, image_idx + "_segmentation.png")
        mask = read_image(mask_path)
        if mask is not None: 
          obj_ids = torch.unique(mask)
          obj_ids = obj_ids[1:]
          masks = mask == obj_ids[:, None, None]
          boxes = masks_to_boxes(masks)

        return image, boxes

        
if __name__ == "__main__":
    train = MoleData("/content/drive/MyDrive/Colab Notebooks/ISIC-2017-DATA/ISIC-2017_Training_Data", "/content/drive/MyDrive/Colab Notebooks/ISIC-2017-DATA/ISIC-2017_Training_Part1_GroundTruth","/content/drive/MyDrive/Colab Notebooks/ISIC-2017-DATA/ISIC-2017_Training_Part3_GroundTruth.csv")
    image, bbox = train[0]
    image = np.array(image.detach().cpu())
    plt.imshow(image.transpose((1,2,0)))
    bbox = bbox[0]
    plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],facecolor='none', edgecolor='b'))
    plt.show()
    print(bbox)
    print(bbox[0], bbox[3], bbox[1] - bbox[0], bbox[3]-bbox[2])