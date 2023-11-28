import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import cv2
import numpy as np

class ISICDataset(Dataset):
    """Dataset for YOLO model."""

    def __init__(self, image_dir, mask_dir, labels, image_size):
        """
        image_dir = folder path to images
        mask_dir = folder path to masks
        labels = csv file of labels
        image_size = size images need to be
        """
        self.size = image_size
        self.images = []
        for filename in os.listdir(image_dir):
          result = filename.endswith('.jpg')
          if result:
            self.images.append(filename)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.labels = pd.read_csv(labels)
        self.samples = []
        for i in range(len(self.images)):
          sample = self.load_samples(i)
          self.samples.append(sample)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.samples[idx]

    def load_samples(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_filename = self.images[idx]
        img_name = os.path.join(self.image_dir, img_filename)
        image = cv2.imread(img_name)

        mk_filename = img_filename.replace('.jpg', '_segmentation.png')
        mk_name = os.path.join(self.mask_dir, mk_filename)
        mask = cv2.imread(mk_name)

        #resize image and mask
        image = cv2.resize(image, (self.size, self.size))
        mask = cv2.resize(mask, (self.size, self.size))

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
        vector = np.array([x + (w/2), y + (h/2), w, h, prob, label1, label2], dtype=int)

        #Convert image to tensor
        image = image.transpose((2, 0, 1)).astype(np.float32)
        image = image/255
        sample = (torch.from_numpy(image), torch.from_numpy(vector))

        return sample