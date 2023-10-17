import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.io import read_image
import os


train_path = "ISIC2018_Task1-2_Training_Input_x2"
seg_path = "ISIC2018_Task1_Training_GroundTruth_x2"


def transform(name):
    if name == 'train':
        return transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                 transforms.Resize((256,256), antialias=True),
                ])
    elif name == 'seg':
        return transforms.Compose(
        	  [
                transforms.Resize((256,256), antialias=True),
              ])

class ISICDataset(Dataset):
    def __init__(self, img_dir, segment_dir=None, train_transform=None, seg_transform=None):
        self.img_dir = img_dir
        self.segment_dir = segment_dir
        
		#Get all the images in each directory
        self.image_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        self.image_files.sort()
        if segment_dir:
            self.seg_files = [f for f in os.listdir(segment_dir) if f.endswith('_segmentation.png')]
            self.seg_files.sort()
      
        
		#Form pairs between the training input images and their ground truths
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
        image = read_image(self.pairs[idx][0]).float()  # Load the image
        image = self.transform(image)	
        if self.segment_dir:
            segmentation = read_image(self.pairs[idx][1]).long()/255
            segmentation = self.target_transform(segmentation)
            return image, segmentation
        else:
            return image
