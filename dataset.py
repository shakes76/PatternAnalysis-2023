import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.io import read_image
import os


train_path = "ISIC2018_Task1-2_Training_Input_x2"
seg_path = "ISIC2018_Task1_Training_GroundTruth_x2"


def transform(name: str) -> transforms.Compose:
    """ Retrieves the relevant transform specific for the set of data presented to it.

        Parameters:
            name: Either train or seg which is relevant to the paths that the dataloader will use.

        Returns:
            A composed transform for the data.
    """
    if name == 'train':
        return transforms.Compose(
                [transforms.Resize((256,256), antialias=True), # Resizes the image to a convenient format.
                ])
    elif name == 'seg':
        return transforms.Compose(
        	  [
                transforms.Resize((256,256), antialias=True),
              ])
    else:
        return transforms.Compose([])

class CustomISICDataset(Dataset):
    def __init__(self, img_dir, segment_dir=None, train_transform=None, seg_transform=None):
        """ Initialises the dataset made for the ISIC data.

            Parameters:
                img_dir: The path for the actual images
                segment_dir: The path for their ground truths
                train_transform: The transform for the training directory
                seg_transform: The transform for the ground truth directory
        """
        self.img_dir = img_dir
        self.segment_dir = segment_dir
        
		#Get all the images in each directory
        self.image_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        self.image_files.sort()
        if segment_dir:
            self.seg_files = [f for f in os.listdir(segment_dir) if f.endswith('_segmentation.png')]
            self.seg_files.sort()
      
        
		# Form pairs between the training input images and their ground truths
        if segment_dir:
            self.pairs = []
            for img_file, seg_file in zip(self.image_files, self.seg_files):
                img_path = os.path.join(img_dir, img_file)
                seg_path = os.path.join(segment_dir, seg_file)
                self.pairs.append((img_path, seg_path))
        
        self.transform = train_transform
        self.target_transform = seg_transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image = read_image(self.pairs[idx][0]).float()/255  # Load the image, divide by 255 have the tensor between 0 and 1
        image = self.transform(image)	
        if self.segment_dir:
            segmentation = read_image(self.pairs[idx][1]).long()/255 # Load the segment
            segmentation = self.target_transform(segmentation)
            return image, segmentation
        else:
            return image
