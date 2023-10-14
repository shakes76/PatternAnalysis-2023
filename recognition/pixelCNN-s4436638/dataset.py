import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode
from torchvision.io import read_image
import glob

class GetADNITrain(Dataset):
    def __init__(self, path, train_split=0.9, train=True):
        
        # Get the list of images and sort them
        # glob.glob gives a list of files in a folder
        # sorted sorts this in alphabetical/numerical order
        image_list = sorted(glob.glob(path)) # [21520, path_string]

        # Shrink to the desired train_split and if in validation or train
        temp_arr_len = int(len(image_list)) # 21520
        if train: # We want to use 90% of train images for train
            image_list = image_list[:int(temp_arr_len * train_split)]
        else: # We want to use 10% of train images for validation
            image_list = image_list[int(temp_arr_len * train_split):]

        # Get the array length
        self.arrLen = len(image_list)
        self.image_list = image_list

    def __getitem__(self, index):

        # Load the image in grayscale [0, 255]
        x = read_image(self.image_list[index], ImageReadMode.GRAY)

        # Normalize between [0, 1]
        x = x / torch.max(x)

        return x
    
    def __len__(self):
        return self.arrLen

class GetADNITest(Dataset):
    def __init__(self, path):
        
        # Get the list of images and sort them
        self.image_list = sorted(glob.glob(path))

        # Get the array length
        self.arrLen = len(self.image_list)

    def __getitem__(self, index):
        # Load the image in grayscale
        x = read_image(self.image_list[index], ImageReadMode.GRAY)

        # Normalize between [0, 1]
        x = x / torch.max(x)

        return x
    
    def __len__(self):
        return self.arrLen