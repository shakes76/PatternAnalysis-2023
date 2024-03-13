import os
import sys
import numpy as np
import skimage.draw
import pandas as pd
from mrcnn.mrcnn.config import Config
from mrcnn.mrcnn.utils import *
from mrcnn.mrcnn.model import * #This will fail as mrcnn is retarded

from utils import get_data_dir


# Root directory of the project
class ISICConfig(Config):
    NAME = "ISIC"
    IMAGES_PER_GPU = 2

    NUM_CLASSES = 1 + 3 # Background + 3 classes for lesions: (melanoma, seborrheic keratosis and benign / Uknown)

    STEPS_PER_EPOCH = 100

    DETECTION_MIN_CONFIDENCE = 0.7


class ISICDataset(Dataset):
    "Should load a subset of our dataset, such that the same class can be used for loading training and validation dataset"

    def load_isic(self, dataset_dir, subset):
        self.add_class("ISIC", 0, "melanoma")
        self.add_class("ISIC", 1, 'seborrheic_keratosis')
        self.add_class("ISIC", 2, "benign / Unknown")
        
        assert subset in ["train", "val"]
        if subset == "labels":
            #Load a df and save it
            self.labels = pd.read_csv(get_data_dir() / dataset_dir/ "")
        dataset_dir = os.path.join(dataset_dir, subset)

        for filename in os.listdir(dataset_dir):
            print(filename)
            #Get the size for each picture
            image_path = os.join(get_data_dir() / dataset_dir, filename)
            height, width = skimage.io.imread(image_path).shape[:2]

            self.add_image("ISIC", 
                           image_id=filename, 
                           image_path=image_path,
                           width=width,
                           height=height,
                           polygons=None)
                           


    def load_mask(self, image_id):
        #Should return an array of masks, one for each instance present in the image
        
        #Note: specifically for the ISIC dataset there is always only one instances per image
        image_info = self.image_info[image_id]


if __name__ == "__main__":
    dset = ISICDataset()
    dset.load_isic(get_data_dir() / "data" / "ISIC", "train")

