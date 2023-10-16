import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import pandas as pd
import os
import cv2
import numpy as np

class ISICDatset(Dataset):
    """Dataset for YOLO model."""

    def __init__(self, image_dir, mask_dir, labels):
        """
        image_dir = folder path to images
        mask_dir = folder path to masks
        labels = csv file of labels
        """
        self.images = []
        for filename in os.listdir(image_dir):
          result = filename.endswith('.jpg')
          if result:
            self.images.append(filename)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.labels = pd.read_csv(labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_filename = self.images[idx]
        img_name = os.path.join(self.image_dir, img_filename)
        image = cv2.imread(img_name)

        mk_filename = img_filename.replace('.jpg', '_segmentation.png')
        mk_name = os.path.join(self.mask_dir, mk_filename)
        mask = cv2.imread(mk_name)

        #get bounding box from mask
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        cntrs = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
        cntr = cntrs[0]
        x,y,w,h = cv2.boundingRect(cntr)

        #Getting probabilities of either label
        label1, label2 = self.labels.iloc[idx, 1:]
        prob = label1 + label2
        #Putting together answer vector
        vector = np.array([prob, x, y, w, h, label1, label2], dtype=int)
        sample = (image, torch.from_numpy(vector))

        return sample