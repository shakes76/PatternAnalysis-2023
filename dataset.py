import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.io import read_image
import os

train_path = "/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Training_Input_x2"
seg_path = "/home/groups/comp3710/ISIC2018/ISIC2018_Task1_Training_GroundTruth_x2"



transform_train = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomCrop(32, padding=4, padding_mode='reflect'),]
)

transform_test = transforms.Compose(
	  [
          
      ])

class ISICDataLoader(Dataset):
    def __init__(self, img_dir, segment_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.segment_dir = segment_dir
        
		#Get all the images in each directory
        self.image_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        self.seg_files = [f for f in os.listdir(segment_dir) if f.endswith('_segmentation.png')]
        self.image_files.sort()
        self.seg_files.sort()

		#Form pairs between the training input images and their ground truths
        self.pairs = []
        for img_file, seg_file in zip(self.image_files, self.seg_files):
            img_path = os.path.join(img_dir, img_file)
            seg_path = os.path.join(segment_dir, seg_file)
            self.pairs.append((img_path, seg_path))
        
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image = read_image(self.pairs[idx][0]).squeeze().float()
        segmentation = read_image(self.pairs[idx][1]).squeeze().float()
        if self.transform:
            image = self.transform(image)
        if self.target_transform:	
            segmentation = self.target_transform(segmentation)
        return image, segmentation


train_loader = ISICDataLoader(train_path, seg_path, transform_train, transform_test)

image, seg = train_loader.__getitem__(1)
print(image.size())
print(seg.size())