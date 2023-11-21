import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode
from torchvision.io import read_image
import glob

class GetADNITrain(Dataset):
    def __init__(self, path, train_split=0.9, train=True):
        
        image_list = []

        # Get the list of images and sort them
        # glob.glob gives a list of files in a folder
        # sorted sorts this in alphabetical/numerical order
        image_list_AD = sorted(glob.glob(path + "AD/*"))
        image_list_NC = sorted(glob.glob(path + "NC/*"))

        # Shrink to the desired train_split and if in validation or train
        temp_arr_len_AD = int(len(image_list_AD))
        temp_arr_len_NC = int(len(image_list_NC))
        if train: # We want to use 90% of train images for train
            image_list += image_list_AD[:int(temp_arr_len_AD * train_split)]
            image_list += image_list_NC[:int(temp_arr_len_NC * train_split)]
        else: # We want to use 10% of train images for validation
            image_list += image_list_AD[int(temp_arr_len_AD * train_split):]
            image_list += image_list_NC[int(temp_arr_len_NC * train_split):]

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
        # glob.glob gives a list of files in a folder
        # sorted sorts this in alphabetical/numerical order
        image_list_AD = sorted(glob.glob(path + "AD/*"))
        image_list_NC = sorted(glob.glob(path + "NC/*"))
        # Get the list of images and sort them
        self.image_list = image_list_AD + image_list_NC

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